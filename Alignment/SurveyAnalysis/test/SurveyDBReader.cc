#include "TFile.h"
#include "TTree.h"
#include "Math/EulerAngles.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/SurveyAnalysis/test/SurveyDBReader.h"

SurveyDBReader::SurveyDBReader(const edm::ParameterSet& cfg):
  theFileName( cfg.getParameter<std::string>("fileName") ),
  theFirstEvent(true)
{
}

void SurveyDBReader::analyze(const edm::Event&, const edm::EventSetup& setup)
{
  typedef AlignTransform SurveyValue;
  typedef Alignments     SurveyValues;

  if (theFirstEvent) {

    edm::ESHandle<SurveyValues> valuesHandle;
    edm::ESHandle<SurveyErrors> errorsHandle;
    
    setup.get<TrackerSurveyRcd>().get(valuesHandle);
    setup.get<TrackerSurveyErrorExtendedRcd>().get(errorsHandle);
    
    const std::vector<SurveyValue>& values = valuesHandle->m_align;
    const std::vector<SurveyError>& errors = errorsHandle->m_surveyErrors;
    
    unsigned int size = values.size();
    
    if ( errors.size() != size )
      {
	edm::LogError("SurveyDBReader")
	  << "Value and error records have different sizes. Abort!";
	
	return;
      }
    
    uint8_t  type = 0;
    uint32_t id   = 0;
    
    ROOT::Math::Cartesian3D<double>* pos(0); // pointer required by ROOT
    ROOT::Math::EulerAngles* rot(0); // pointer required by ROOT
    const align::ErrorMatrix* cov(0); // pointer required by ROOT
    
    TFile fout(theFileName.c_str(), "RECREATE");
    TTree tree("survey", "");
    
    tree.Branch("type", &type, "type/b");
    tree.Branch("id",   &id,   "id/i");
    tree.Branch("pos", "ROOT::Math::Cartesian3D<double>",    &pos);
    tree.Branch("rot", "ROOT::Math::EulerAngles",   &rot);
    tree.Branch("cov", "align::ErrorMatrix", &cov);
    
    for (unsigned int i = 0; i < size; ++i)
      {
	const SurveyValue& value = values[i];
	const SurveyError& error = errors[i];
	
	const CLHEP::Hep3Vector&  transl = value.translation();
	const CLHEP::HepRotation& orient = value.rotation();
	const align::ErrorMatrix& matrix = error.matrix();

	ROOT::Math::Cartesian3D<double> coords( transl.x(),transl.y(),transl.z() );
	ROOT::Math::EulerAngles angles( orient.phi(), orient.theta(), orient.psi() );
	
	type = error.structureType();
	id   = error.rawId();
	pos  = &coords;
	rot  = &angles;
	cov  = &matrix;
	
	tree.Fill();
      }
    
    fout.Write();

    edm::LogInfo("SurveyDBReader")
      << "Number of alignables read " << size << std::endl;

    theFirstEvent = false;
  }
}
