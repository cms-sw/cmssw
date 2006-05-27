#ifndef EgammaElectronAlgos_SiStripElectronAlgo_h
#define EgammaElectronAlgos_SiStripElectronAlgo_h
// -*- C++ -*-
//
// Package:     EgammaElectronAlgos
// Class  :     SiStripElectronAlgo
// 
/**\class SiStripElectronAlgo SiStripElectronAlgo.h RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:11:58 EDT 2006
// $Id$
//

// system include files

// user include files

// forward declarations

class SiStripElectronAlgo
{

   public:
      SiStripElectronAlgo();
      virtual ~SiStripElectronAlgo();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      SiStripElectronAlgo(const SiStripElectronAlgo&); // stop default

      const SiStripElectronAlgo& operator=(const SiStripElectronAlgo&); // stop default

      // ---------- member data --------------------------------

};


#endif
