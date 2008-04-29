#ifndef ThePEGSource_h
#define ThePEGSource_h

/** \class ThePEGSource
 *  $Id: ThePEGSource.h 93 2008-02-25 20:15:36Z stober $
 *  
 *  Oliver Oberst <oberst@ekp.uni-karlsruhe.de>
 *  Fred-Markus Stober <stober@ekp.uni-karlsruhe.de>
 */

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "ThePEG/Repository/EventGenerator.h"
#include <string>

#include "HepMC/IO_BaseClass.h"

namespace edm
{
	class ThePEGSource : public GeneratedInputSource
	{
	public:
		ThePEGSource(const ParameterSet &, const InputSourceDescription &);
		virtual ~ThePEGSource();
	private:
		virtual bool produce(Event &);
		virtual void endRun(Run &run);
		ThePEG::EGPtr eg_;

		HepMC::IO_BaseClass *iobc_;
		void InitRepository(const ParameterSet &) const;
		void InitGenerator(const ParameterSet &);
	};
} 

#endif
