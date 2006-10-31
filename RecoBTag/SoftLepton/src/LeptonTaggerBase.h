#ifndef RecoBTag_SoftLepton_LeptonTaggerBase_h
#define RecoBTag_SoftLepton_LeptonTaggerBase_h

/**  \class LeptonTaggerBase
 *
 *   Abstract class for lepton b-tagging a jet, based on track-to-jet parameters
 *
 *   $Date: 2006/10/22 16:30:48 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class TVector3;
class reco::SoftLeptonProperties;

class LeptonTaggerBase {
public:
  LeptonTaggerBase (void) {}
  virtual ~LeptonTaggerBase (void) {}
 
  /// calculate a jet b-tagging discriminant based on track-to-jet parameters:
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const = 0;

};

#endif // RecoBTag_SoftLepton_LeptonTaggerBase_h
