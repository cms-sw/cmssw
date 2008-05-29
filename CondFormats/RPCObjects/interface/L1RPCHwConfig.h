#ifndef CondFormats_RPCObjects_L1RPCHwConfig_h
#define CondFormats_RPCObjects_L1RPCHwConfig_h
// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCHwConfig
// 
/**\class L1RPCHwConfig L1RPCHwConfig.h CondFormats/RPCObjects/interface/L1RPCHwConfig.h

 Description: Contains multiple BX triggering info. Also info about disabled devices

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Apr  9 13:48:06 CEST 2008
// $Id: L1RPCHwConfig.h,v 1.2 2008/04/10 13:39:12 fruboes Exp $
//

// system include files

// user include files

// forward declarations
#include <set>
#include <vector>

#include <iostream>

struct L1RPCDevCoords {
 public:
  L1RPCDevCoords(): m_tower(0), m_PAC(255) {};
  L1RPCDevCoords(int tower, int sector, int segment): m_tower(tower), m_PAC(sector*12+segment) {};
  int getTower() {return m_tower;};
  int getPAC() {return m_PAC;};
  int getSector() {return m_PAC/12;};
  int getSegment() {return m_PAC%12;};
  bool operator() (const L1RPCDevCoords & l1, const L1RPCDevCoords & l2 ) const{
    if (l1.m_tower != l2.m_tower)
       return l1.m_tower < l2.m_tower;
    else
       return l1.m_PAC < l2.m_PAC;
  }; 

 private:
// Type "a" is not supported ( CORAL : "AttributeList" from "CoralBase" ) 
//   signed char m_tower;
//   unsigned char m_PAC;
   char m_tower;
   char m_PAC;

};


class L1RPCHwConfig
{

   public:
      L1RPCHwConfig();
      virtual ~L1RPCHwConfig();

      bool isActive(int tower, int sector, int segment) const
           {
             return m_disabledDevices.end()==m_disabledDevices.find( L1RPCDevCoords(tower, sector, segment) ); 
           };

      bool isActive(int tower, int PAC) const
           {
             return m_disabledDevices.end()==m_disabledDevices.find( L1RPCDevCoords(tower, PAC/12, PAC%12) ); 
           };

      void enablePAC(int tower, int sector, int segment, bool enable);

      void enableTower(int tower, bool enable);

      void enableTowerInCrate(int tower, int crate, bool enable);

      void enableCrate(int logSector, bool enable); // logSector == crate number /// enables one crate, all towers

      void enableAll(bool enable);

      int size() const {return m_disabledDevices.size(); } ;

      int getFirstBX() const {return m_firstBX;};
      int getLastBX() const {return m_lastBX;};

      void setFirstBX(int bx) { m_firstBX = bx;};
      void setLastBX(int bx) {  m_lastBX = bx;};
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------


   private:

      int m_firstBX;
      int m_lastBX;
      std::set<L1RPCDevCoords,L1RPCDevCoords > m_disabledDevices;

      // ---------- member data --------------------------------

};


#endif
