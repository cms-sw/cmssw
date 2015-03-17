#ifndef FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H
#define FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_rule.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/phoenix/bind/bind_member_function.hpp>

#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <functional>
#include <unordered_map>

struct BinaryOP;
struct UnaryOP;
struct Nil {};

struct ExpressionAST
{

    typedef
        boost::variant<
          Nil,
          int,
          std::string,
          boost::recursive_wrapper<ExpressionAST>,
          boost::recursive_wrapper<BinaryOP>,
          boost::recursive_wrapper<UnaryOP>
        >
    Type;

    ExpressionAST():
        expr(Nil())
    {
    }

    template <typename Expr>
    ExpressionAST(Expr const& expr):
        expr(expr)
    {
    }

    int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const;

    ExpressionAST& operator!();

    Type expr;
};

ExpressionAST operator>(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator>=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator==(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator<=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator<(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator!=(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator&(ExpressionAST const& lhs, ExpressionAST const& rhs);
ExpressionAST operator|(ExpressionAST const& lhs, ExpressionAST const& rhs);

struct BinaryOP
{
    enum class OP
    {
        GREATER, GREATER_EQUAL, EQUAL, LESS_EQUAL, LESS, NOT_EQUAL, AND, OR
    } op;
    ExpressionAST left;
    ExpressionAST right;

    BinaryOP(OP op, ExpressionAST const& left, ExpressionAST const& right):
        op(op),
        left(left),
        right(right)
    {
    }

    int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const
    {
        switch(op)
        {
            case OP::GREATER:
                return left.evaluate(detId,trackerTopology) > right.evaluate(detId,trackerTopology);
            case OP::GREATER_EQUAL:
                return left.evaluate(detId,trackerTopology) >= right.evaluate(detId,trackerTopology);
            case OP::EQUAL:
                return left.evaluate(detId,trackerTopology) == right.evaluate(detId,trackerTopology);
            case OP::LESS_EQUAL:
                return left.evaluate(detId,trackerTopology) <= right.evaluate(detId,trackerTopology);
            case OP::LESS:
                return left.evaluate(detId,trackerTopology) < right.evaluate(detId,trackerTopology);
            case OP::NOT_EQUAL:
                return left.evaluate(detId,trackerTopology) != right.evaluate(detId,trackerTopology);
            case OP::AND:
                return left.evaluate(detId,trackerTopology) && right.evaluate(detId,trackerTopology);
            case OP::OR:
                return left.evaluate(detId,trackerTopology) || right.evaluate(detId,trackerTopology);
        }
        return 0;
    }
};

struct UnaryOP
{
     enum class OP
     {
         NEG
     } op;
     ExpressionAST subject;
     UnaryOP(OP op, ExpressionAST const& subject):
         op(op),
         subject(subject)
     {
     }

     int evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const
     {
        switch (op)
        {
            case OP::NEG:
                return !subject.evaluate(detId,trackerTopology);
        }
        return 0;
     }

};

struct WalkAST
{
    typedef void result_type;

    void operator()(boost::spirit::qi::info::nil) const {}
    void operator()(int n) const { std::cout << n; }
    void operator()(std::string str) const { std::cout << str; }
    void operator()(ExpressionAST const& ast) const
    {
        boost::apply_visitor(*this, ast.expr);
    }

    void operator()(BinaryOP const& expr) const
    {
        std::cout << "(";
        boost::apply_visitor(*this, expr.left.expr);
        switch(expr.op)
        {
            case BinaryOP::OP::GREATER:
                std::cout<<" > ";
                break;
            case BinaryOP::OP::GREATER_EQUAL:
                std::cout<<" >= ";
                break;
            case BinaryOP::OP::EQUAL:
                std::cout<<" == ";
                break;
            case BinaryOP::OP::LESS_EQUAL:
                std::cout<<" <= ";
                break;
            case BinaryOP::OP::LESS:
                std::cout<<" < ";
                break;
            case BinaryOP::OP::NOT_EQUAL:
                std::cout<<" != ";
                break;
            case BinaryOP::OP::AND:
                std::cout<<" & ";
                break;
            case BinaryOP::OP::OR:
                std::cout<<" | ";
                break;
        }
        boost::apply_visitor(*this, expr.right.expr);
        std::cout << ')';
    }

    void operator()(UnaryOP const& expr) const
    {
        switch (expr.op)
        {
            case UnaryOP::OP::NEG:
                std::cout<<" !(";
                break;
        }
        boost::apply_visitor(*this, expr.subject.expr);
        std::cout << ')';
    }
};


class TrackerDetIdSelector
{
    private:
        const DetId& _detId;
        const TrackerTopology& _trackerTopology;



    public:
        typedef std::function<int(const TrackerTopology& trackerTopology, const DetId&)> DetIdFunction;
        typedef std::unordered_map<std::string, DetIdFunction> StringFunctionMap;
        const static StringFunctionMap functionTable;
    
        TrackerDetIdSelector(const DetId& detId, const TrackerTopology& trackerTopology):
            _detId(detId),
            _trackerTopology(trackerTopology)
        {
            namespace qi = boost::spirit::qi;
            namespace ascii = boost::spirit::ascii;
            namespace phoenix = boost::phoenix;
        }

        bool passSelection(std::string selectionStr) const
        {
            namespace qi = boost::spirit::qi;
            namespace ascii = boost::spirit::ascii;
            namespace phoenix = boost::phoenix;

            std::string::const_iterator begin = selectionStr.cbegin();
            std::string::const_iterator end = selectionStr.cend();

            qi::rule<std::string::const_iterator, std::string(), ascii::space_type>
                identifierFctRule =qi::lexeme[+qi::alpha[qi::_val+=qi::_1]];

            qi::rule<std::string::const_iterator, ExpressionAST(), ascii::space_type>
                identifierRule =
                    '!' >> identifierRule[qi::_val=!qi::_1] |
                    (qi::true_[qi::_val=1] | qi::false_[qi::_val=0]) |
                    (qi::int_[qi::_val=qi::_1]) |
                    identifierFctRule[qi::_val=qi::_1];

            qi::rule<std::string::const_iterator, ExpressionAST(), ascii::space_type>
                expressionRule =
                    (identifierRule >> qi::lit(">") >> identifierRule)[qi::_val=qi::_1>qi::_2] |
                    (identifierRule >> qi::lit(">=") >> identifierRule)[qi::_val=qi::_1>=qi::_2] |
                    (identifierRule >> qi::lit("<") >> identifierRule)[qi::_val=qi::_1<qi::_2] |
                    (identifierRule >> qi::lit("<=") >> identifierRule)[qi::_val=qi::_1<=qi::_2] |
                    (identifierRule >> qi::lit("==") >> identifierRule)[qi::_val=qi::_1==qi::_2] |
                    (identifierRule >> qi::lit("!=") >> identifierRule)[qi::_val=qi::_1!=qi::_2];

            qi::rule<std::string::const_iterator, ExpressionAST(), ascii::space_type,  qi::locals<ExpressionAST>>
                comboRule =
                    ('(' >> comboRule[qi::_a=qi::_1] >> ')' >>
                        *(qi::lit("&") >> '(' >> comboRule[qi::_a=qi::_a & qi::_1] >> ')' |
                          qi::lit("|") >> '(' >> comboRule[qi::_a=qi::_a | qi::_1] >> ')'))[qi::_val=qi::_a] |
                    expressionRule[qi::_val=qi::_1];
                    
                    
            ExpressionAST exprAST;

            bool success = qi::phrase_parse(begin,end, comboRule, ascii::space, exprAST);
            if (!success || begin!=end)
            {
                std::cout<<"error while parsing:"<<std::endl;
                for (auto it=selectionStr.cbegin(); it!=begin; ++it)
                {
                    std::cout << *it;
                }
                std::cout << "^^^";
                for (auto it=begin; it!=selectionStr.cend(); ++it)
                {
                    std::cout << *it;
                }
                std::cout<<std::endl;
            }
            WalkAST walker;
            walker(exprAST);
            std::cout<<std::endl;
            return exprAST.evaluate(_detId,_trackerTopology);
        }




        unsigned int pxbLayer(const DetId &id) const
        {
            return 1;
        }
        /*
        unsigned int tobLayer(const DetId &id) const
        {
        }

        unsigned int tibLayer(const DetId &id) const
        {
        }

        unsigned int pxbLadder(const DetId &id) const
        {
        }

        unsigned int pxbModule(const DetId &id) const
        {
        }

        unsigned int pxfModule(const DetId &id) const
        {
        }

        unsigned int tobModule(const DetId &id) const
        {
        }

        unsigned int tecModule(const DetId &id) const
        {
        }

        unsigned int tibModule(const DetId &id) const
        {
        }

        unsigned int tidModule(const DetId &id) const
        {
        }

        unsigned int tobSide(const DetId &id) const
        {
        }

        unsigned int tecSide(const DetId &id) const
        {
        }

        unsigned int tibSide(const DetId &id) const
        {
        }

        unsigned int tidSide(const DetId &id) const
        {
        }

        unsigned int pxfSide(const DetId &id) const
        {
        }

        unsigned int tobRod(const DetId &id) const
        {
        }

        unsigned int tecWheel(const DetId &id) const
        {
        }

        unsigned int tidWheel(const DetId &id) const
        {
        }

        unsigned int tecOrder(const DetId &id) const
        {
        }

        unsigned int tibOrder(const DetId &id) const
        {
        }

        unsigned int tidOrder(const DetId &id) const
        {
        }

        unsigned int tecRing(const DetId &id) const
        {
        }

        unsigned int tidRing(const DetId &id) const
        {
        }

        unsigned int tecPetalNumber(const DetId &id) const
        {
        }


        bool tobIsDoubleSide(const DetId &id) const
        {
        }

        bool tecIsDoubleSide(const DetId &id) const
        {
        }

        bool tibIsDoubleSide(const DetId &id) const
        {
        }

        bool tidIsDoubleSide(const DetId &id) const
        {
        }

        bool tobIsZPlusSide(const DetId &id) const
        {
        }

        bool tobIsZMinusSide(const DetId &id) const
        {
        }

        bool tibIsZPlusSide(const DetId &id) const
        {
        }

        bool tibIsZMinusSide(const DetId &id) const
        {
        }

        bool tidIsZPlusSide(const DetId &id) const
        {
        }

        bool tidIsZMinusSide(const DetId &id) const
        {
        }

        bool tecIsZPlusSide(const DetId &id) const
        {
        }

        bool tecIsZMinusSide(const DetId &id) const
        {
        }

        bool tobIsStereo(const DetId &id) const
        {
        }

        bool tecIsStereo(const DetId &id) const
        {
        }

        bool tibIsStereo(const DetId &id) const
        {
        }

        bool tidIsStereo(const DetId &id) const
        {
        }

        uint32_t tobStereo(const DetId &id) const
        {
        }

        uint32_t tibStereo(const DetId &id) const
        {
        }

        uint32_t tidStereo(const DetId &id) const
        {
        }

        uint32_t tecStereo(const DetId &id) const
        {
        }

        uint32_t tibGlued(const DetId &id) const
        {
        }

        uint32_t tecGlued(const DetId &id) const
        {
        }

        uint32_t tobGlued(const DetId &id) const
        {
        }

        uint32_t tidGlued(const DetId &id) const
        {
        }

        bool tobIsRPhi(const DetId &id) const
        {
        }

        bool tecIsRPhi(const DetId &id) const
        {
        }

        bool tibIsRPhi(const DetId &id) const
        {
        }

        bool tidIsRPhi(const DetId &id) const
        {
        }

        bool tecIsBackPetal(const DetId &id) const
        {
        }

        bool tecIsFrontPetal(const DetId &id) const
        {
        }

        unsigned int tibString(const DetId &id) const
        {
        }


        bool tibIsInternalString(const DetId &id) const
        {
        }

        bool tibIsExternalString(const DetId &id) const
        {
        }

        bool tidIsBackRing(const DetId &id) const
        {
        }

        bool tidIsFrontRing(const DetId &id) const
        {
        }

        unsigned int pxfDisk(const DetId &id) const
        {
        }

        unsigned int pxfBlade(const DetId &id) const
        {
        }

        unsigned int pxfPanel(const DetId &id) const
        {
        }
    */

};


class Accessor:
    public boost::static_visitor<int>
{
    private:
        const DetId& _detId;
        const TrackerTopology& _trackerTopology;
    public:
        Accessor(const DetId& detId, const TrackerTopology& trackerTopology):
            _detId(detId),
            _trackerTopology(trackerTopology)
        {
        }

        int operator()(Nil i) const
        {
            throw "should not happen";
        }
        int operator()(const int& i) const
        {
            return i;
        }
        int operator()(const std::string& s) const
        {
            TrackerDetIdSelector::StringFunctionMap::const_iterator it = TrackerDetIdSelector::functionTable.find(s);
            int value = 0;
            if (it != TrackerDetIdSelector::functionTable.cend())
            {
                value = (it->second)(_trackerTopology,_detId);
                std::cout<<"attr="<<s<<", value="<<value<<std::endl;
            }
            else
            {
                std::cout<<"attr="<<s<<" unknown"<<std::endl;
            }
            return value;
        }
        int operator()(const ExpressionAST& ast) const
        {
            return ast.evaluate(_detId,_trackerTopology);
        }
        int operator()(const BinaryOP& binaryOP) const
        {
            return binaryOP.evaluate(_detId,_trackerTopology);
        }
        int operator()(const UnaryOP& unaryOP) const
        {
            return unaryOP.evaluate(_detId,_trackerTopology);
        }
};

int ExpressionAST::evaluate(const DetId& detId, const TrackerTopology& trackerTopology) const
{
    return boost::apply_visitor(Accessor(detId,trackerTopology),this->expr);
}

#endif
