/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include "TFile.h"

/**
 *\brief Repeats the RP factorization.
 **/
class RedoAlignmentFactorization : public edm::EDAnalyzer
{
  public:
    RedoAlignmentFactorization(const edm::ParameterSet &ps); 
    ~RedoAlignmentFactorization() {}

  private:
    edm::ParameterSet ps;

    virtual void beginJob();
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es) {}
    virtual void endJob() {}
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

RedoAlignmentFactorization::RedoAlignmentFactorization(const ParameterSet &_ps) : ps(_ps)
{
}

//----------------------------------------------------------------------------------------------------

void RedoAlignmentFactorization::beginJob()
{
  TFile *tdf = new TFile(ps.getUntrackedParameter<string>("taskDataFile", "task_data.root").c_str());
  AlignmentGeometry *g = (AlignmentGeometry *) tdf->Get("geometry");
  if (!g) {
    AlignmentTask *t = (AlignmentTask *) tdf->Get("task");
    g = &t->geometry;
  }

  RPAlignmentCorrections input(ps.getUntrackedParameter<string>("inputFile"));

  bool equalWeights = ps.getUntrackedParameter<bool>("equalWeights");
  unsigned int verbosity = ps.getUntrackedParameter<unsigned int>("verbosity", 1);
  string expandedFileName = ps.getUntrackedParameter<string>("expandedFileName");
  string factoredFileName = ps.getUntrackedParameter<string>("factoredFileName");

  RPAlignmentCorrections expanded, factored;
  input.FactorRPFromSensorCorrections(expanded, factored, *g, equalWeights, verbosity);

  if (!expandedFileName.empty())
    expanded.WriteXMLFile(expandedFileName);
  
  if (!factoredFileName.empty())
    factored.WriteXMLFile(factoredFileName);
 
  delete tdf; 
}

DEFINE_FWK_MODULE(RedoAlignmentFactorization);

