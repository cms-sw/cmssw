#ifndef CSCSegment_CSCSegmentBuilderPluginFactory_h
#define CSCSegment_CSCSegmentBuilderPluginFactory_h

/** \class CSCSegmentBuilderPluginFactory
 *  Plugin factory for concrete CSCSegmentBuilder algorithms
 *
 * $Date: 2006/04/03 10:10:10 $
 * $Revision: 1.2 $
 * \author M. Sani
 * 
 */

#include <PluginManager/PluginFactory.h>
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
