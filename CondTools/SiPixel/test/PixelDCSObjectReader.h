#ifndef CondTools_SiPixel_PixelDCSObjectReader_h
#define CondTools_SiPixel_PixelDCSObjectReader_h

/** \class PixelDCSObjectReader
 *
 *  Class to dump Pixel DCS object from database to ROOT file.
 *
 *  $Date: 2008/11/30 19:41:09 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "TTree.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

template <class Record>
class PixelDCSObjectReader:
  public edm::EDAnalyzer
{
  typedef typename Record::Object Object;

  public:

  PixelDCSObjectReader( const edm::ParameterSet& ) {}

  void analyze( const edm::Event&, const edm::EventSetup& );
};

template <class Record>
void PixelDCSObjectReader<Record>::analyze(const edm::Event&,
                                           const edm::EventSetup& setup)
{
  edm::ESHandle<Object> handle;

  setup.get<Record>().get(handle);

  Object object = *handle;

  edm::Service<TFileService> fs;

  TTree* tree = fs->make<TTree>("tree", "");

  tree->Branch("object", &object);
  tree->Fill();
}

#endif
