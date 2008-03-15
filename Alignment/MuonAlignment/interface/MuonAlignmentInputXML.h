#ifndef Alignment_MuonAlignment_MuonAlignmentInputXML_h
#define Alignment_MuonAlignment_MuonAlignmentInputXML_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputXML
// 
/**\class MuonAlignmentInputXML MuonAlignmentInputXML.h Alignment/MuonAlignment/interface/MuonAlignmentInputXML.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Mon Mar 10 16:37:55 CDT 2008
// $Id$
//

// system include files
#include <string>
#include <xercesc/dom/DOMElement.hpp>

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

// forward declarations

class MuonAlignmentInputXML: public MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputXML(std::string fileName);
      virtual ~MuonAlignmentInputXML();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

   private:
      MuonAlignmentInputXML(const MuonAlignmentInputXML&); // stop default

      const MuonAlignmentInputXML& operator=(const MuonAlignmentInputXML&); // stop default

      void recursiveGetId(std::map<unsigned int, Alignable*> &alignableNavigator, const std::vector<Alignable*> &alignables) const;

      void fillAliToIdeal(std::map<Alignable*, Alignable*> &alitoideal, const std::vector<Alignable*> alignables, const std::vector<Alignable*> ideals) const;

      Alignable *getNode(std::map<unsigned int, Alignable*> &alignableNavigator, const xercesc_2_7::DOMElement *node) const;
      Alignable *getDTnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const xercesc_2_7::DOMElement *node) const;
      Alignable *getCSCnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const xercesc_2_7::DOMElement *node) const;

      double parseDouble(const XMLCh *str, const char *attribute) const;
      void set_one_position(Alignable *ali, const align::PositionType &pos, const align::RotationType &rot) const;

      void do_setposition (const xercesc_2_7::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_setape      (const xercesc_2_7::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_setsurveyerr(const xercesc_2_7::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;

      // ---------- member data --------------------------------
      std::string m_fileName;

      XMLCh *str_operation;
      XMLCh *str_collection;
      XMLCh *str_name;
      XMLCh *str_DTBarrel;
      XMLCh *str_DTWheel;
      XMLCh *str_DTStation;
      XMLCh *str_DTChamber;
      XMLCh *str_DTSuperLayer;
      XMLCh *str_DTLayer;
      XMLCh *str_CSCEndcap;
      XMLCh *str_CSCStation;
      XMLCh *str_CSCRing;
      XMLCh *str_CSCChamber;
      XMLCh *str_CSCLayer;
      XMLCh *str_setposition;
      XMLCh *str_setape;
      XMLCh *str_setsurveyerr;
      XMLCh *str_relativeto;
      XMLCh *str_rawId;
      XMLCh *str_wheel;
      XMLCh *str_station;
      XMLCh *str_sector;
      XMLCh *str_superlayer;
      XMLCh *str_layer;
      XMLCh *str_endcap;
      XMLCh *str_ring;
      XMLCh *str_chamber;
      XMLCh *str_x;
      XMLCh *str_y;
      XMLCh *str_z;
      XMLCh *str_phix;
      XMLCh *str_phiy;
      XMLCh *str_phiz;
      XMLCh *str_alpha;
      XMLCh *str_beta;
      XMLCh *str_gamma;
      XMLCh *str_xx;
      XMLCh *str_xy;
      XMLCh *str_xz;
      XMLCh *str_xa;
      XMLCh *str_xb;
      XMLCh *str_xc;
      XMLCh *str_yy;
      XMLCh *str_yz;
      XMLCh *str_ya;
      XMLCh *str_yb;
      XMLCh *str_yc;
      XMLCh *str_zz;
      XMLCh *str_za;
      XMLCh *str_zb;
      XMLCh *str_zc;
      XMLCh *str_aa;
      XMLCh *str_ab;
      XMLCh *str_ac;
      XMLCh *str_bb;
      XMLCh *str_bc;
      XMLCh *str_cc;
      XMLCh *str_none;
      XMLCh *str_ideal;
      XMLCh *str_container;
      XMLCh *str_decimalpoint;
      XMLCh *str_exponent;
      XMLCh *str_EXPONENT;
};


#endif
