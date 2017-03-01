#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHOniaToMuMuBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHOniaToMuMuBuilder_h
/** \class BPHOniaToMuMuBuilder
 *
 *  Description: 
 *     Class to build Psi(1,2) and Upsilon(1,2,3) candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

class BPHMuonPtSelect;
class BPHMuonEtaSelect;
class BPHChi2Select;
class BPHMassSelect;
class BPHRecoSelect;
class BPHMomentumSelect;
class BPHVertexSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHOniaToMuMuBuilder {

 public:

  enum oniaType { Phi , Psi1, Psi2, Ups , Ups1, Ups2, Ups3 };

  /** Constructor
   */
  BPHOniaToMuMuBuilder(
    const edm::EventSetup& es,
    const BPHRecoBuilder::BPHGenericCollection* muPosCollection,
    const BPHRecoBuilder::BPHGenericCollection* muNegCollection );

  /** Destructor
   */
  virtual ~BPHOniaToMuMuBuilder();

  /** Operations
   */
  /// build resonance candidates
  std::vector<BPHPlusMinusConstCandPtr> build();

  /// extract list of candidates of specific type
  /// candidates are rebuilt applying corresponding mass constraint
  std::vector<BPHPlusMinusConstCandPtr> getList( oniaType type,
                                        BPHRecoSelect    * dSel = 0,
                                        BPHMomentumSelect* mSel = 0,
                                        BPHVertexSelect  * vSel = 0,
                                        BPHFitSelect     * kSel = 0 );

  /// retrieve original candidate from a copy with the same daughters
  /// obtained through "getList"
  BPHPlusMinusConstCandPtr getOriginalCandidate( 
                           const BPHRecoCandidate& cand );

  /// set cuts
  void setPtMin  ( oniaType type, double pt  );
  void setEtaMax ( oniaType type, double eta );
  void setMassMin( oniaType type, double m   );
  void setMassMax( oniaType type, double m   );
  void setProbMin( oniaType type, double p   );
  void setConstr ( oniaType type, double mass, double sigma );

  /// get current cuts
  double getPtMin  ( oniaType type ) const;
  double getEtaMax ( oniaType type ) const;
  double getMassMin( oniaType type ) const;
  double getMassMax( oniaType type ) const;
  double getProbMin( oniaType type ) const;
  double getConstrMass ( oniaType type ) const;
  double getConstrSigma( oniaType type ) const;

 private:

  // private copy and assigment constructors
  BPHOniaToMuMuBuilder           ( const BPHOniaToMuMuBuilder& x );
  BPHOniaToMuMuBuilder& operator=( const BPHOniaToMuMuBuilder& x );

  std::string muPosName;
  std::string muNegName;

  const edm::EventSetup* evSetup;
  const BPHRecoBuilder::BPHGenericCollection* posCollection;
  const BPHRecoBuilder::BPHGenericCollection* negCollection;

  struct OniaParameters {
    BPHMuonPtSelect *   ptSel;
    BPHMuonEtaSelect*  etaSel;
    BPHMassSelect   * massSel;
    BPHChi2Select   * chi2Sel;
    double mass;
    double sigma;
    bool updated;
  };
  bool updated;

  std::map< oniaType, OniaParameters > oniaPar;
  std::map< oniaType, std::vector<BPHPlusMinusConstCandPtr> > oniaList;
  std::vector<BPHPlusMinusConstCandPtr> fullList;

  void setNotUpdated();
  void setParameters( oniaType type,
                      double ptMin, double etaMax,
                      double massMin, double massMax,
                      double probMin,
                      double mass, double sigma );
  void extractList( oniaType type );

};


#endif

