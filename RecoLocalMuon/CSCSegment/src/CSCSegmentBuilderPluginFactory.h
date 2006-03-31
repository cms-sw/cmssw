#ifndef CSCSegment_CSCSegmentBuilderPluginFactory_h
#define CSCSegment_CSCSegmentBuilderPluginFactory_h

/** \class CSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete CSCSegmentBuilder algorithms
 *
 */

#include <PluginManager/PluginFactory.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentAlgorithm.h>

class edm::ParameterSet;

class CSCSegmentBuilderPluginFactory : public seal::PluginFactory<CSCSegmentAlgorithm *(const edm::ParameterSet&)>{
public:
    CSCSegmentBuilderPluginFactory();
    static CSCSegmentBuilderPluginFactory* get (void);

private:
    static CSCSegmentBuilderPluginFactory s_instance;
};
#endif
