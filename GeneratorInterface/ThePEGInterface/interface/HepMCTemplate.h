#ifndef GeneratorInterface_ThePEGInterface_HepMCTemplate_h
#define GeneratorInterface_ThePEGInterface_HepMCTemplate_h

/** \class HepMCTemplate
 *  
 * @brief Header file defines template struct needed for CMSSW to convert HepMC file
 *
 */


#include <ThePEG/Vectors/HepMCTraits.h>

namespace ThePEG {

	template<> struct HepMCTraits<HepMC::GenEvent> :
		public HepMCTraitsBase<
			HepMC::GenEvent, HepMC::GenParticle,
			HepMC::GenVertex, HepMC::Polarization,
			HepMC::PdfInfo> {};

}

#endif // GeneratorInterface_ThePEGInterface_HepMCTemplate_h
