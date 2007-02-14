#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "TVector3.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"
#include "RecoBTag/SoftLepton/src/ElectronTaggerMLP.h"

/** \class ElectronTagger
 *
 *
 *  $Id: ElectronTagger.h,v 1.1 2007/02/14 16:37:53 demine Exp $
 *  $Date: 2007/02/14 16:37:53 $
 *  $Revision: 1.1 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public LeptonTaggerBase
{

public:

  ElectronTagger() : theBTagNN() {}
  virtual ~ElectronTagger() {}

  /// b-tag a jet based on track-to-jet parameters:
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const 
  {
    return theBTagNN.value(0, properties.ptRel, properties.sip3d, properties.deltaR, properties.ratioRel);
  }

private:

  mutable ElectronTaggerMLP theBTagNN;

};

#endif // RecoBTag_SoftLepton_ElectronTagger_h
