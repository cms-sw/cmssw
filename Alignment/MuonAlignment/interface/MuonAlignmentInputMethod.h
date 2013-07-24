#ifndef Alignment_MuonAlignment_MuonAlignmentInputMethod_h
#define Alignment_MuonAlignment_MuonAlignmentInputMethod_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputMethod
// 
/**\class MuonAlignmentInputMethod MuonAlignmentInputMethod.h Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 14:10:22 CST 2008
// $Id: MuonAlignmentInputMethod.h,v 1.2 2008/03/20 21:39:26 pivarski Exp $
//

// system include files
#include <boost/shared_ptr.hpp>

#include "FWCore/Framework/interface/EventSetup.h"

// user include files
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

// forward declarations

class MuonAlignmentInputMethod {
   public:
      MuonAlignmentInputMethod();
      virtual ~MuonAlignmentInputMethod();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

   protected:
      boost::shared_ptr<DTGeometry> idealDTGeometry(const edm::EventSetup &iSetup) const;
      boost::shared_ptr<CSCGeometry> idealCSCGeometry(const edm::EventSetup &iSetup) const;

   private:
      MuonAlignmentInputMethod(const MuonAlignmentInputMethod&); // stop default

      const MuonAlignmentInputMethod& operator=(const MuonAlignmentInputMethod&); // stop default

      // ---------- member data --------------------------------
};


#endif
