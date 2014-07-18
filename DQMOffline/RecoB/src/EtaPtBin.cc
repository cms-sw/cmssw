#include "DQMOffline/RecoB/interface/EtaPtBin.h"

#include <algorithm>
#include <sstream>




EtaPtBin::EtaPtBin ( const bool& etaActive_ , const double& etaMin_ , const double& etaMax_ ,
		     const bool& ptActive_  , const double& ptMin_  , const double& ptMax_ )
  : etaActive ( etaActive_ ) , etaMin ( etaMin_ ) , etaMax ( etaMax_ ) ,
    ptActive  (  ptActive_ ) , ptMin  (  ptMin_ ) , ptMax  (  ptMax_ )   {

  descriptionString = buildDescriptionString ( etaActive , etaMin , etaMax ,
					       ptActive  , ptMin  , ptMax  );
}



std::string EtaPtBin::buildDescriptionString
		( const bool& etaActive_ , const double& etaMin_ , const double& etaMax_ ,
		  const bool& ptActive_  , const double& ptMin_  , const double& ptMax_)
{
  // create string only from the active parts
  std::stringstream stream ( "" );

  if ( etaActive_ ) {
    stream << "_ETA_" << etaMin_ << "-" << etaMax_;
  }

  if ( ptActive_ ) {
    stream << "_PT_" << ptMin_ << "-" << ptMax_;
  }
  if (!(etaActive_||ptActive_)) stream << "_GLOBAL";

  std::string descr(stream.str());
  // remove blanks which are introduced when adding doubles
  std::remove(descr.begin(), descr.end(), ' ');
  std::replace(descr.begin(), descr.end(), '.' , 'v' );

  return descr;
}

bool EtaPtBin::inBin(const reco::Jet & jet, const double jec) const
{
  return inBin(jet.eta(), jet.pt()*jec);
}

// bool EtaPtBin::inBin(const BTagMCTools::JetFlavour & jetFlavour) const
// {
//   return inBin(jetFlavour.underlyingParton4Vec().Eta(),
// 	       jetFlavour.underlyingParton4Vec().Pt());
// }


bool EtaPtBin::inBin (const double & eta , const double & pt ) const {
  if ( etaActive ) {
    if ( fabs(eta) < etaMin ) return false;
    if ( fabs(eta) > etaMax ) return false;
  }

  if ( ptActive ) {
    if ( pt < ptMin ) return false;
    if ( pt > ptMax ) return false;
  }

  return true;
}
