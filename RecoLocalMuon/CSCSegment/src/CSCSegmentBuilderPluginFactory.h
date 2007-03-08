#ifndef CSCSegment_CSCSegmentBuilderPluginFactory_h
#define CSCSegment_CSCSegmentBuilderPluginFactory_h

/** \class CSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete CSCSegmentBuilder algorithms
 *
 * $Date: 2006/05/08 17:45:31 $
 * $Revision: 1.3 $
 * \author M. Sani
 * 
 */

#include <FWCore/PluginManager/interface/PluginFactory.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>

class edm::ParameterSet;

class CSCSegmentBuilderPluginFactory : public seal::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)>{
public:
    /// Constructor
    CSCSegmentBuilderPluginFactory();
    
    static CSCSegmentBuilderPluginFactory* get (void);

private:

    static CSCSegmentBuilderPluginFactory s_instance;
};
#endif
