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
// $Id: MuonAlignmentInputXML.h,v 1.8 2010/07/30 04:20:13 pivarski Exp $
//

// system include files
#include <string>
#include "xercesc/dom/DOMElement.hpp"
#include "xercesc/util/XercesDefs.hpp"


// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

// forward declarations

class MuonAlignmentInputXML: public MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputXML(const std::string& fileName);
      ~MuonAlignmentInputXML() override;

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const override;

   private:
      MuonAlignmentInputXML(const MuonAlignmentInputXML&) = delete; // stop default

      const MuonAlignmentInputXML& operator=(const MuonAlignmentInputXML&) = delete; // stop default

      void recursiveGetId(std::map<unsigned int, Alignable*> &alignableNavigator, const align::Alignables &alignables) const;

      void fillAliToIdeal(std::map<Alignable*, Alignable*> &alitoideal, const align::Alignables& alignables, const align::Alignables& ideals) const;

      Alignable *getNode(std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node, const AlignableObjectId&) const;
      Alignable *getDTnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node, const AlignableObjectId&) const;
      Alignable *getCSCnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node, const AlignableObjectId&) const;

      double parseDouble(const XMLCh *str, const char *attribute) const;
      void set_one_position(Alignable *ali, const align::PositionType &pos, const align::RotationType &rot) const;

      void do_setposition (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_setape      (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_setsurveyerr(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_moveglobal  (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_movelocal   (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_rotatelocal (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_rotatebeamline (const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;
      void do_rotateglobalaxis(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const;

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
      XMLCh *str_moveglobal;
      XMLCh *str_movelocal;
      XMLCh *str_rotatelocal;
      XMLCh *str_rotatebeamline;
      XMLCh *str_rotateglobalaxis;
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
      XMLCh *str_axisx;
      XMLCh *str_axisy;
      XMLCh *str_axisz;
      XMLCh *str_angle;
      XMLCh *str_x;
      XMLCh *str_y;
      XMLCh *str_z;
      XMLCh *str_phix;
      XMLCh *str_phiy;
      XMLCh *str_phiz;
      XMLCh *str_alpha;
      XMLCh *str_beta;
      XMLCh *str_gamma;
      XMLCh *str_rphi;
      XMLCh *str_phi;
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
};


#endif
