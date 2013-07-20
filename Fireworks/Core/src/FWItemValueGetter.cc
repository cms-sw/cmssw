// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemValueGetter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Nov 30 16:15:43 EST 2008
// $Id: FWItemValueGetter.cc,v 1.11 2012/12/02 09:49:59 amraktad Exp $
//

// system include files
#include <sstream>
#include <cstdio>
#include "TMath.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWItemValueGetter.h"

#include "Fireworks/Core/interface/FWExpressionEvaluator.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
#include "CommonTools/Utils/src/Grammar.h"
#include "CommonTools/Utils/interface/Exception.h"

#include "Fireworks/Core/src/expressionFormatHelpers.h"


//==============================================================================
//==============================================================================
//==============================================================================


FWItemValueGetter::FWItemValueGetter(const edm::TypeWithDict& iType, const std::string& iPurpose):
   m_type(iType),
   m_titleWidth(0)
{
   if (!strcmp(iType.name().c_str(), "CaloTower"))
   {
      if ( iPurpose == "ECal" )
         addEntry("emEt", 1, "et", "GeV");
      else if ( iPurpose == "HCal" )
         addEntry("hadEt", 1, "et", "GeV");
      else if (iPurpose == "HCal Outer")
         addEntry("outerEt", 1, "et", "GeV");
   }
   else if (strstr(iPurpose.c_str(), "Beam Spot") )
   {
      addEntry("x0", 2, "x", "cm");
      addEntry("y0", 2, "y", "cm");
      addEntry("z0", 2, "z", "cm");
   }
   else if (strstr(iPurpose.c_str(), "Conversion") )
   {
      addEntry("pairMomentum().rho()", 1, "pt", "GeV" );
      addEntry("pairMomentum().eta()", 2, "eta");
      addEntry("pairMomentum().phi()", 2, "phi");
   }
   else if (strstr(iPurpose.c_str(), "Candidate") || strstr(iPurpose.c_str(), "GenParticle"))
   {
      addEntry("pdgId()", 0, "pdg");
      bool x = addEntry("pt", 1);
      if (!x) x = addEntry("et", 1);
      if (!x) addEntry("energy", 1);
   }
   else if (iPurpose == "Jets" )
   {
      addEntry("et", 1);
   }
   else {
      // by the default  add pt, et, or energy
      bool x = addEntry("pt", 1);
      if (!x) x = addEntry("et", 1);
      if (!x) addEntry("energy", 1);
   }

   if (addEntry("eta", 2))
      addEntry("phi",  2);
}



bool FWItemValueGetter::addEntry(std::string iExpression, int iPrec, std::string iTitle, std::string iUnit)
{
   using namespace boost::spirit::classic;

   reco::parser::ExpressionPtr tmpPtr;
   reco::parser::Grammar grammar(tmpPtr, m_type);

   if(m_type != edm::TypeWithDict() && iExpression.size()) 
   {
      using namespace fireworks::expression;

      //Backwards compatibility with old format
      std::string temp = oldToNewFormat(iExpression);

      //now setup the parser
      try 
      {
         if(parse(temp.c_str(), grammar.use_parser<1>() >> end_p, space_p).full) 
         {
            m_entries.push_back(Entry(tmpPtr, iExpression, iUnit, iTitle.empty() ? iExpression :iTitle , iPrec));
            m_titleWidth = TMath::Max(m_titleWidth, (int) m_entries.back().m_title.size());
            return true;
         }
      } 
      catch(const reco::parser::BaseException& e)
      {
         // std::cout <<"failed to parse "<<iExpression<<" because "<<reco::parser::baseExceptionWhat(e)<<std::endl;
      }
   }
   return false;
}


//______________________________________________________________________________

double
FWItemValueGetter::valueFor(const void* iObject, int idx) const
{
   //  std::cout << " value for " << idx << "size " <<  m_entries.size() <<std::endl;
   edm::ObjectWithDict o(m_type, const_cast<void *>(iObject));
   return m_entries[idx].m_expr->value(o);
}

UInt_t
FWItemValueGetter::precision(int idx) const
{
   return m_entries[idx].m_precision;
}

std::vector<std::string> 
FWItemValueGetter::getTitles() const
{
   std::vector<std::string> titles;
   titles.reserve(m_entries.size());

   for (std::vector<Entry >::const_iterator i  = m_entries.begin() ; i != m_entries.end(); ++i) 
      titles.push_back((*i).m_title.empty() ? (*i).m_expression : (*i).m_title );

   return titles;
}

int 
FWItemValueGetter::numValues() const
{
   return static_cast<int>(m_entries.size());
}
//______________________________________________________________________________

const std::string& 
FWItemValueGetter::getToolTip(const void* iObject) const
{
   static std::string buff(128, 0);
   static std::string fs = "\n %*s = %.*f";

   edm::ObjectWithDict o(m_type, const_cast<void *>(iObject));

   int off = 0;
   for ( std::vector<Entry >::const_iterator i = m_entries.begin() ; i != m_entries.end(); ++i) {
      const Entry& e = *i;
      off += snprintf(&buff[off], 127, fs.c_str(), m_titleWidth, e.m_title.c_str(),  e.m_precision ? (e.m_precision+1) : 0,  e.m_expr->value(o));
   }

   // std::cout << buff;
   return buff;
}

