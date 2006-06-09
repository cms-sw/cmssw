#ifndef RPCTrigger_RPCCurl_h
#define RPCTrigger_RPCCurl_h

/** \class RPCCurl
 *
 * \brief Class describng 2PI "rings" constructed from RpcDetId's of the same eta (same as L1RpcRoll in ORCA)
 * \author Tomasz Fruboes
 * \todo Implement & use global roll no
 *
 */

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"


class RPCCurl {
  public:
    
    enum { IROLL_MAX = 17, NPOS = 3, NHPLANES  = 6 };  
    
    
    
    struct stripCords {
      uint32_t detRawId;
      int stripNo;
      bool isVirtual;
    };
    
    struct stripCordsOp{
      bool operator()(const stripCords sc1, const stripCords sc2) const
      {
        if (sc1.detRawId!=sc2.detRawId)
          return sc1.detRawId<sc2.detRawId;
        else
          return sc1.stripNo<sc2.stripNo;
      }
    };
    
    struct RPCConnection {
      int PAC;
      int tower;
      int logplane;
      int posInCone;
    };
    
    typedef std::vector<RPCConnection> RPCConnectionsVec;
    typedef std::map<stripCords, RPCConnectionsVec, stripCordsOp> RPCLinks;
    
  public:
    RPCCurl();
    ~RPCCurl();
    
    bool addDetId(RPCDetInfo detInfo);
    int makeRefConnections(RPCCurl *);
    int makeOtherConnections(float phiCenter);

    void printContents();
    
    int getMinTower() const;
    int getMaxTower() const;
    int getCurlId() const;
    bool isRefPlane() const;
    
  private:
    void setRefPlane();
    void doVirtualStrips();
    int giveLogPlaneForTower(int tower);
    
    void updatePhiStripsMap(RPCDetInfo detInfo);//
        
  private:
    int m_towerMin; ///< The lowest tower no. to which curl contributes
    int m_towerMax; ///< The highest tower no. to which curl contributes
    int m_hwPlane; ///< Hardware plane no.
    int m_region; ///< Region no - 0 for barell +-1 for endcaps
    int m_ring;  ///< Wheel number for barell, ring number for endcaps
    int m_roll;  ///< roll no
    int m_curlId;///< this curlId
    int m_physStripsInCurl; ///< Number of existing strips in curl;
    int m_virtStripsInCurl; ///< Number of virtual strips in curl;
    
    int m_globRoll;
    
    bool m_isDataFresh; ///< Defines if data has real world contents
    bool m_isRefPlane;  ///< tells if detIds from this curl form a reference plane
    bool m_didVirtuals;
    RPCLinks m_links;
    
    
    
    typedef std::map<uint32_t, RPCDetInfo> RPCDetInfoMap;
    RPCDetInfoMap m_RPCDetInfoMap; ///< Stores all DetId`s of a curl
    
    //typedef std::map<float, uint32_t> RPCDetInfoPhiMap;
    typedef std::map<float, uint32_t> RPCDetInfoPhiMap;
    RPCDetInfoPhiMap m_RPCDetPhiMap; ///< Stores DetId`s rawId in a phi order.
           
    /// \todo offset should be defined elswhere
    struct phiMapCompare
    {
      bool operator()(const float f1, const float f2) const /// put strips of phi from 0 to 5 degrees at the end
      {
        const float pi = 3.141592654;
        const float offset = (5./360.)*2*pi;
        float ff1, ff2;
        ff1=f1;
        ff2=f2;
        
        if (ff1 < offset)
          ff1+=2*pi;
        if (ff2 < offset)
          ff2+=2*pi;
        
        return ff1 < ff2;
        
      }
    };
    
    typedef std::map<float, stripCords, phiMapCompare> GlobalStripPhiMap;
    GlobalStripPhiMap m_stripPhiMap;    
        
    static const int mrtow [RPCCurl::IROLL_MAX+1] [RPCCurl::NHPLANES] [RPCCurl::NPOS];
    static const int mrlogp [RPCCurl::IROLL_MAX+1] [RPCCurl::NHPLANES] [RPCCurl::NPOS];
    static const unsigned int LOGPLANE_SIZE[17][6];
    //static const unsigned short int stripsToAdd[RPCCurl::IROLL_MAX+1];



};
#endif
