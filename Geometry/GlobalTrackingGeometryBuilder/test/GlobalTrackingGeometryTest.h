#ifndef GlobalTrackingGeometryTest_h
#define GlobalTrackingGeometryTest_h

/*
 * \class GlobalTrackingGeometryTest
 *
 * EDAnalyzer to test the GlobalTrackingGeometry.
 *
 *  $Date: 2006/07/25 09:47:59 $
 *  $Revision: 1.1 $
 *  \author M. Sani
 */

class CSCGeometry;
class DTGeometry;    
class RPCGeometry;
class TrackerGeometry;
class GlobalTrackingGeometry;
    
class GlobalTrackingGeometryTest : public edm::EDAnalyzer {

public:
 
    explicit GlobalTrackingGeometryTest( const edm::ParameterSet& );
    ~GlobalTrackingGeometryTest();

    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    void analyzeCSC(const GlobalTrackingGeometry* geo, const CSCGeometry* cscGeometry);
    void analyzeDT(const GlobalTrackingGeometry* geo, const DTGeometry* dtGeometry);
    void analyzeRPC(const GlobalTrackingGeometry* geo, const RPCGeometry* rpcGeometry);
    void analyzeTracker(const GlobalTrackingGeometry* geo, const TrackerGeometry* tkGeometry);
         
    const std::string& myName() { return my_name; }

private: 
    
    std::string my_name;    

};

#endif
