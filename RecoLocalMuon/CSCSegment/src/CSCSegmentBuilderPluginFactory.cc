/** \file CSCSegmentBuilderPluginFactory.cc
 *
 * $Date: 2006/04/03 10:10:10 $
 * $Revision: 1.2 $
 * \author M. Sani
 * 
 */

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

CSCSegmentBuilderPluginFactory CSCSegmentBuilderPluginFactory::s_instance;

CSCSegmentBuilderPluginFactory::CSCSegmentBuilderPluginFactory () : 
  seal::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)>("CSCSegmentBuilderPluginFactory"){}

CSCSegmentBuilderPluginFactory* CSCSegmentBuilderPluginFactory::get (){
  
  return &s_instance; 
}

