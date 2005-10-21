// -*- C++ -*-
//
// Package:     <package>
// Module:      cutParser
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris D Jones
// Created:     Sun Aug  7 20:45:55 EDT 2005
// $Id$
//
// Revision history
//
// $Log$

// system include files
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <vector>

// user include files
#include "PhysicsTools/CandUtils/interface/cutParser.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

//
// constants, enums and typedefs
//
using aod::Candidate;
using aod::Selector;

namespace aod {

typedef boost::spirit::scanner<const char*, boost::spirit::scanner_policies_t > ScannerUsed;
typedef boost::spirit::rule<ScannerUsed> Rule_t;


struct ExpressionBase {
   virtual ~ExpressionBase() {};
   virtual double value(const Candidate&) const = 0;
};
struct ExpressionNumber : public ExpressionBase {
   virtual double value(const Candidate&) const {
      return m_value;
   }
   double m_value;
   ExpressionNumber( double iValue ) : m_value(iValue) {//std::cout << m_value <<std::endl;
   }
};

//class Candidate;
//typedef std::map<std::string, double> Candidate;

struct ExpressionVar : public ExpressionBase {
   ExpressionVar(PCandMethod iMethod ):
    m_method(iMethod) { //std::cout << m_var<<std::endl;
   }
   
   virtual double value(const Candidate& iCand) const {
     return (iCand.*m_method)();
   }
  PCandMethod m_method;
};

typedef std::vector<boost::shared_ptr<ExpressionBase> > ExpressionStack;
class ExpressionNumberSetter {
public:
   ExpressionNumberSetter( ExpressionStack& iStack) : stack_(iStack){}
   
   void operator()( double iNumber) const {
      stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionNumber( iNumber ) ) );
   }
private:
   ExpressionStack& stack_;
};

class ExpressionVarSetter {
public:
  ExpressionVarSetter( ExpressionStack& iStack,
		       const CandidateMethods& iMethods) : 
    stack_(iStack),methods_(iMethods){}
   
   void operator()( const char* iVarStart, const char* iVarEnd) const {
     std::string methodName( iVarStart, iVarEnd);
     methodName.erase( methodName.find_last_of(' '), methodName.size() );
     CandidateMethods::const_iterator itMethod = methods_.find(methodName);
     if( itMethod == methods_.end() ) {
       throw edm::Exception(edm::errors::Configuration, std::string("unknown method name \""+methodName+"\"") );
     }
     stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionVar( itMethod->second ) ) );
   }
private:
   ExpressionStack& stack_;
   const CandidateMethods& methods_;
};


struct ComparisonBase {
   virtual bool compare( double, double ) const = 0;
};

template<class CompT>
struct Comparison : public ComparisonBase {
   virtual bool compare( double iLHS, double iRHS ) const {
      CompT comp;
      return comp( iLHS, iRHS );
   }
};

typedef std::vector<boost::shared_ptr<ComparisonBase> > ComparisonStack;
template< class CompT>
class ComparisonSetter {
public:
   ComparisonSetter( ComparisonStack& iStack) : stack_(iStack){}
   
   void operator()( const char ) const {
      stack_.push_back( boost::shared_ptr<ComparisonBase>( new Comparison<CompT>() ) );
   }
private:
   ComparisonStack& stack_;
};


typedef std::vector<boost::shared_ptr<Selector> > SelectorStack;

struct BinarySelector : public Selector {
   BinarySelector( boost::shared_ptr<ExpressionBase> iLHS,
                   boost::shared_ptr<ComparisonBase> iComp,
                   boost::shared_ptr<ExpressionBase> iRHS ) :
   m_lhs( iLHS), m_comp(iComp), m_rhs( iRHS ) {}
  virtual bool operator()( const Candidate& iCand) const {
    return m_comp->compare( m_lhs->value(iCand), m_rhs->value(iCand) );
   }
   boost::shared_ptr<ExpressionBase> m_lhs;
   boost::shared_ptr<ComparisonBase> m_comp;
   boost::shared_ptr<ExpressionBase> m_rhs;
};

class BinarySelectorSetter {
public:
   BinarySelectorSetter( SelectorStack& iSelStack,
                         ComparisonStack& iCompStack, ExpressionStack& iExpStack) : 
   sStack_(iSelStack), cStack_(iCompStack), eStack_(iExpStack){}
   
   void operator()( const char* iVarStart, const char* iVarEnd) const {
      boost::shared_ptr<ExpressionBase> rhs = eStack_.back(); eStack_.pop_back();
      boost::shared_ptr<ExpressionBase> lhs = eStack_.back(); eStack_.pop_back();
      boost::shared_ptr<ComparisonBase> comp = cStack_.back(); cStack_.pop_back();
      sStack_.push_back( boost::shared_ptr<Selector>( new BinarySelector( lhs, comp, rhs ) ) );
   }
private:
   SelectorStack& sStack_;
   ComparisonStack& cStack_;
   ExpressionStack& eStack_;
};


struct TrinarySelector : public Selector {
   TrinarySelector( boost::shared_ptr<ExpressionBase> iLHS,
                   boost::shared_ptr<ComparisonBase> iComp1,
                   boost::shared_ptr<ExpressionBase> iMid,
                    boost::shared_ptr<ComparisonBase> iComp2,
                    boost::shared_ptr<ExpressionBase> iRHS) :
   m_lhs( iLHS), m_comp1(iComp1), m_mid(iMid), m_comp2(iComp2),m_rhs( iRHS ) {}
  virtual bool operator()( const Candidate& iCand ) const {
      return m_comp1->compare( m_lhs->value(iCand), m_mid->value(iCand) ) &&
             m_comp2->compare( m_mid->value(iCand), m_rhs->value(iCand) );
   }
   boost::shared_ptr<ExpressionBase> m_lhs;
   boost::shared_ptr<ComparisonBase> m_comp1;
   boost::shared_ptr<ExpressionBase> m_mid;
   boost::shared_ptr<ComparisonBase> m_comp2;
   boost::shared_ptr<ExpressionBase> m_rhs;
};


class TrinarySelectorSetter {
public:
   TrinarySelectorSetter( SelectorStack& iSelStack,
                         ComparisonStack& iCompStack, ExpressionStack& iExpStack) : 
   sStack_(iSelStack), cStack_(iCompStack), eStack_(iExpStack){}
   
   void operator()( const char* iVarStart, const char* iVarEnd) const {
      boost::shared_ptr<ExpressionBase> rhs = eStack_.back(); eStack_.pop_back();
      boost::shared_ptr<ExpressionBase> mid = eStack_.back();eStack_.pop_back();
      boost::shared_ptr<ExpressionBase> lhs = eStack_.back();eStack_.pop_back();
      boost::shared_ptr<ComparisonBase> comp2 = cStack_.back();cStack_.pop_back();
      boost::shared_ptr<ComparisonBase> comp1 = cStack_.back();cStack_.pop_back();
      sStack_.push_back( boost::shared_ptr<Selector>( new TrinarySelector( lhs, comp1, mid, comp2, rhs ) ) );
   }
private:
   SelectorStack& sStack_;
   ComparisonStack& cStack_;
   ExpressionStack& eStack_;
};



enum Combiner { kAnd, kOr };

typedef std::vector< Combiner > CombinerStack;


struct AndCombiner : public Selector {
   AndCombiner( boost::shared_ptr<Selector> iLHS,
                boost::shared_ptr<Selector> iRHS ) :
   m_lhs( iLHS), m_rhs( iRHS ) {}
   virtual bool operator()( const Candidate & iCand ) const {
      return (*m_lhs)(iCand) && (*m_rhs)(iCand);
   }
   boost::shared_ptr<Selector> m_lhs;
   boost::shared_ptr<Selector> m_rhs;
};

struct OrCombiner : public Selector {
   OrCombiner( boost::shared_ptr<Selector> iLHS,
               boost::shared_ptr<Selector> iRHS ) :
   m_lhs( iLHS), m_rhs( iRHS ) {}
   virtual bool operator()(const Candidate& iCand) const {
      return (*m_lhs)(iCand) || (*m_rhs)(iCand);
   }
boost::shared_ptr<Selector> m_lhs;
boost::shared_ptr<Selector> m_rhs;
};

struct CombinerSetter {
   CombinerSetter( Combiner iCombiner, CombinerStack& iStack ):
   m_comb( iCombiner), m_stack(iStack) {}
   
   void operator()(const char ) const {
      m_stack.push_back( m_comb );
   }
   Combiner m_comb;
   CombinerStack& m_stack;
};


struct CutSetter {
   CutSetter( boost::shared_ptr<Selector>& iCut,
              SelectorStack& iSelStack,
              CombinerStack& iCombStack) :
   m_cut( iCut ), m_sStack(iSelStack), m_cStack(iCombStack) {}
   
   void operator()(const char*, const char* ) const {
      if( 0 == m_cut.get() ) {
         m_cut = m_sStack.back();
         m_sStack.pop_back();
      } else {
         if( m_cStack.back() == kAnd ) {
            boost::shared_ptr<Selector> lhs = m_cut;
            m_cut = boost::shared_ptr<Selector>(new AndCombiner(lhs, m_sStack.back() ) );
         } else {
            boost::shared_ptr<Selector> lhs = m_cut;
            m_cut = boost::shared_ptr<Selector>(new OrCombiner(lhs, m_sStack.back() ) );
         }
         m_cStack.pop_back();
         m_sStack.pop_back();
      }
   }
   boost::shared_ptr<Selector>& m_cut;
   SelectorStack& m_sStack;
   CombinerStack& m_cStack;
};
   
   

bool cutParser( const std::string& iValue,
		const CandidateMethods& iMethods,
		boost::shared_ptr<Selector>& iCut ) {
   
   using namespace boost::spirit;
   ExpressionStack expressionStack;
   ComparisonStack cStack;
   SelectorStack sStack;
   CombinerStack combStack;
   
   Rule_t number = real_p                       [ExpressionNumberSetter(expressionStack)];
   Rule_t var = (alpha_p >> *alnum_p)           [ExpressionVarSetter(expressionStack, iMethods)];
   
   Rule_t comparison_op = ch_p('<')             [ComparisonSetter<std::less<double> >(cStack)]           | 
                         (ch_p('<') >> ch_p('=')[ComparisonSetter<std::less_equal<double> >(cStack)])    | 
                         ch_p('=')              [ComparisonSetter<std::equal_to<double> >(cStack)]       | 
                         (ch_p('>') >> ch_p('=')[ComparisonSetter<std::greater_equal<double> >(cStack)]) | 
                         ch_p('>')              [ComparisonSetter<std::greater<double> >(cStack)]        | 
                         (ch_p('!') >> ch_p('=')[ComparisonSetter<std::not_equal_to<double> >(cStack)]);
   
   Rule_t expression = var | number;
   Rule_t binary_comp = expression >> comparison_op >> expression;
   Rule_t trinary_comp = expression >> comparison_op >> var >> comparison_op >> expression;
   Rule_t logical_combiner = ch_p('&')[CombinerSetter(kAnd, combStack)] | ch_p('|')[CombinerSetter(kOr, combStack)];
   Rule_t cut = (trinary_comp [TrinarySelectorSetter(sStack, cStack, expressionStack)] | 
                 binary_comp  [BinarySelectorSetter(sStack, cStack, expressionStack)]) [CutSetter(iCut, sStack, combStack)] % logical_combiner;
   return parse<>( iValue.c_str(),
		   (
		    cut
		    )
		   ,
		   space_p).full;
}
  
  
}
