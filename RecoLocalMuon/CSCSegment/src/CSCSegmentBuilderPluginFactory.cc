// This is CSCSegmentBuilderPluginFactory.cc

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

CSCSegmentBuilderPluginFactory CSCSegmentBuilderPluginFactory::s_instance;

CSCSegmentBuilderPluginFactory::CSCSegmentBuilderPluginFactory () : 
  seal::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)>("CSCSegmentBuilderPluginFactory"){}

CSCSegmentBuilderPluginFactory* CSCSegmentBuilderPluginFactory::get (){
  
  return &s_instance; 
}

