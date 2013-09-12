#ifndef RecoBTag_SoftLepton_MuonTaggerNoIP_h
#define RecoBTag_SoftLepton_MuonTaggerNoIP_h

#include "TVector3.h"
#include "RecoBTag/SoftLepton/src/LeptonTaggerBase.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerNoIPMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2013/05/20 22:52:25 $
 *   $Revision: 1.3 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTaggerNoIP : public LeptonTaggerBase {
public:
  MuonTaggerNoIP (void) : theNet() {}
  virtual ~MuonTaggerNoIP (void) {}

  /// b-tag a jet based on track-to-jet parameters:
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const 
  {
    return theNet.value( 0, properties.ptRel, properties.ratioRel, properties.deltaR, axis.Mag(), axis.Eta() ) +
           theNet.value( 1, properties.ptRel, properties.ratioRel, properties.deltaR, axis.Mag(), axis.Eta() );
  }

private:
  mutable MuonTaggerNoIPMLP theNet;

};

#endif // RecoBTag_SoftLepton_MuonTaggerNoIP_h
