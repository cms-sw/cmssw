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
// $Id: L1RPCHwConfig.h,v 1.5 2010/02/26 15:50:38 fruboes Exp $
//

// system include files

// user include files

// forward declarations
#include <set>
#include <vector>
#include <sstream>

#include <iostream>

struct L1RPCDevCoords {
 public:
  L1RPCDevCoords(): m_tower(-255), m_PAC(-255) {};
  L1RPCDevCoords(int tower, int sector, int segment): m_tower(tower), m_PAC(sector*12+segment) {};
  int getTower() {return m_tower;};
  int getPAC() {return m_PAC;};
  int getSector() {return m_PAC/12;};
  int getSegment() {return m_PAC%12;};

  std::string toString() const { 
      std::stringstream ss;
      ss << m_tower << " " << m_PAC;  
      return ss.str();
  };

  bool operator<(const L1RPCDevCoords & l2 ) const{
    if (this->m_tower != l2.m_tower)
       return this->m_tower < l2.m_tower;
    return this->m_PAC < l2.m_PAC;
  }

  bool operator==(const L1RPCDevCoords & l2) const{
    return (this->m_tower == l2.m_tower) && (this->m_PAC == l2.m_PAC);
  }
 

 //private:
   signed short m_tower;
   signed short m_PAC;

};


class L1RPCHwConfig
{

   public:
      L1RPCHwConfig();
      virtual ~L1RPCHwConfig();


      
      bool isActive(int tower, int sector, int segment) const
           {
             return m_disabledDevices.find( L1RPCDevCoords(tower, sector, segment) )==m_disabledDevices.end(); 
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
      void dump() const {
         for(std::set<L1RPCDevCoords>::const_iterator it=m_disabledDevices.begin(); it!=m_disabledDevices.end();++it){
           std::cout << it->toString() << std::endl;
         }
      };

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------


   private:

      std::set<L1RPCDevCoords> m_disabledDevices;

      // ---------- member data --------------------------------

};


#endif
