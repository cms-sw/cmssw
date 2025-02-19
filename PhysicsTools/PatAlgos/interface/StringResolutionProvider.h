#ifndef PhysicsTools_PatAlgos_StringResolutionProvider_H
#define PhysicsTools_PatAlgos_StringResolutionProvider_H
#include "DataFormats/PatCandidates/interface/CandKinResolution.h"
#include "PhysicsTools/PatAlgos/interface/KinematicResolutionProvider.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"

/**
   \class   StringResolutionProvider StringResolutionProvider.h "PhysicsTools/PatAlgos/interface/StringResolutionProvider.h"

   \brief   Class to provide resolution factors for PAT candidates

   A class to provide resolution functions for PAT candidates. The class is derived from the 
   KinematicResolutionProvider class. It expects the following parameters:

   * parametrization :   indicates the used representation (e.g. EtEtaPhi). This parameter 
                         is MANDATORY.
   * resolutions     :   a vector of edm::ParameterSets, that contain the resolutions func-
                         tionspotentially in multiple bins of a cetain parameter. For the 
			 default implementeation we have bins of reco::Candidate's eta in 
			 mind, though might be any kind of variable that is accessible as 
			 memeber function of reco::Candidate. This parameter is MANDATORY.

   The edm::ParameterSets of the _resolutions_ parameter are expected to be of form: 

   * bin             :   Indicated the binning compatible with the StringCutObjectSelector. 
                         If omitted no binning is applied and the obtained resolution func-
                         tions are assumed to be valid for all candidates. This parameter 
			 is OPTIONAL.
   * et/eta/phi      :   Contain the parametrization in et if working in the EtEtaPhi pa-
                         rametrization. These parameters are OPTIONAL. They are expected to 
			 be present in the et/eta/phi representation though. 
   * constraint      :   adding further information on constraints. It needs to investigated 
                         further what the meaning of this parameter exactly is before we can 
			 decide about its future. This parameter is OPTIONAL.
   
   We expect that cfi files of this form will be generated automatically. edm::ParameterSets 
   for other represenations will be implemented on request.
*/

class StringResolutionProvider : public KinematicResolutionProvider {

 public:
  /// short cut within the common namespace
  typedef StringObjectFunction<reco::Candidate> Function;

  /// default constructor
  StringResolutionProvider(const edm::ParameterSet& cfg);
  /// default destructor
  virtual ~StringResolutionProvider();
  /// get a CandKinResolution object from the service 
  virtual pat::CandKinResolution getResolution(const reco::Candidate& cand) const;

 private:
  /// a vector of constrtaints for the CanKinResolution 
  /// object
  std::vector<pat::CandKinResolution::Scalar> constraints_;
  /// a parametrization for the CanKinResolution object
  /// (this needs an extension)
  pat::CandKinResolution::Parametrization parametrization_;
  /// a vector of strings for the binning
  std::vector<std::string> bins_;
  /// vectors for the resolution functions
  std::vector<std::string> funcEt_, funcEta_, funcPhi_;
};

#endif
