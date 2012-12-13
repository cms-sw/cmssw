#ifndef Alignment_MuonAlignment_MuonAlignmentInputXML_h
#define Alignment_MuonAlignment_MuonAlignmentInputXML_h
/**\class MuonAlignmentInputXML

 Class for using a custom XML data file as an input for muon alignment

*/
//
// Original Author:  Jim Pivarski
//         Created:  Mon Mar 10 16:37:55 CDT 2008
//
// $Id: MuonAlignmentInputXML.h,v 1.9 2011/06/07 19:28:47 khotilov Exp $
//

#include <string>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/util/XercesDefs.hpp>

#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"


class MuonAlignmentInputXML: public MuonAlignmentInputMethod
{
public:

  typedef XERCES_CPP_NAMESPACE::DOMElement DOMElement;
  typedef std::map<Alignable*, bool> AliSet;
  typedef std::map<unsigned int, Alignable*> AliNavigator;
  typedef std::map<Alignable*, Alignable*> Ali2Ideal;

  MuonAlignmentInputXML(std::string fileName);
  virtual ~MuonAlignmentInputXML();

  virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

private:

  MuonAlignmentInputXML(const MuonAlignmentInputXML&); // stop default

  const MuonAlignmentInputXML& operator=(const MuonAlignmentInputXML&); // stop default

  void recursiveGetId(AliNavigator &alignableNavigator, const align::Alignables &alignables) const;

  void fillAliToIdeal(Ali2Ideal &alitoideal, const align::Alignables alignables, const align::Alignables ideals) const;

  Alignable *getNode(AliNavigator &alignableNavigator, const DOMElement *node) const;
  Alignable *getDTnode(align::StructureType structureType, AliNavigator &alignableNavigator, const DOMElement *node) const;
  Alignable *getCSCnode(align::StructureType structureType, AliNavigator &alignableNavigator, const DOMElement *node) const;

  double parseDouble(const XMLCh *str, const char *attribute) const;
  void set_one_position(Alignable *ali, const align::PositionType &pos, const align::RotationType &rot) const;

  void do_setposition (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_setape      (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_setsurveyerr(const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_moveglobal  (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_movelocal   (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_rotatelocal (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_rotatebeamline (const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;
  void do_rotateglobalaxis(const DOMElement *node, AliSet &aliset, Ali2Ideal &alitoideal) const;

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
