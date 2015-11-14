#ifndef HCALMECOLLECTION_H
#define HCALMECOLLECTION_H

/*
 *	file:			HcalMECollection.h
 *	Author:			Viktor Khristenko
 *	Start Date:		03/04/2015
 *
 *	TODO:
 *		1) Extend MEAxis for variable binning
 *		2) Allow labels to be put for axis
 *		3) Allow some Mes to be updated
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

#include <vector>
#include <string>
#include "boost/ptr_container/ptr_map.hpp"
#include "TH1.h"

namespace hcaldqm
{
	enum CollectionType
	{
		iBooker,
		iGetter,
		nCollectionType
	};

	class MEInfo
	{
		public:
			MEInfo(edm::ParameterSet const& ps);

			void setName(std::string n)				{name=n;}

			std::string const getName() const				{return name;}
			edm::ParameterSet const& getPS() const			{return meps;}

		private:
			std::string					name;
			edm::ParameterSet const&	meps;
	};

	//	Simple struct to detect the type of axis input
	struct MEAxis
	{
		bool edges, wnbins;
		std::string title;
		int nbins;
		double min,max;
		double *bins;
		std::vector<std::string> labels;
		bool log;
	};

/*	struct ObjectBits
	{
		kLogx		= 19,
		kLogy		= 20,
		kLogz		= 21,
	};
	*/

	/*
	 *	HcalMECollection Class
	 *	Access to all Monitor Elements thru a dictionary
	 */
	class HcalMECollection
	{
		public:
			HcalMECollection(edm::ParameterSet const&, int debug=0);
			~HcalMECollection();

			//	Book MEs based on the PSet
			void book(DQMStore::IBooker&, std::string&);

			//	Retrieve MEs based on PSet
			void retrieve(DQMStore::IGetter&, std::string const&);

			//	reset
			void reset(int const periodflag);

			//	Simple getters
			MonitorElement& getME(std::string name) {return (*this)[name];}
			MonitorElement& operator[](std::string);

			//	Update all the MEs
			void update();

		private:
			inline void debug(std::string const msg)
			{
				if (_debug==0)
					return;

				std::cout << "%MSG" << std::endl;
				std::cout << "%MSG-d HCALDQM::" << "HcalMECollection::" << msg;
				std::cout << std::endl;
			}
			inline bool isVPSet(std::string const& name)
			{
				return name[0]=='v';
			}

		private:
			//	do the actual Booking
			void doBook(DQMStore::IBooker&, MEInfo const&);
			void doRetrieve(DQMStore::IGetter&, MEInfo const&);
			void forceEmpty();
			void setBit(TObject* o, int const b)
			{
				o->SetBit(BIT(b));
			}
			bool testBit(TObject* o, int const b)
			{
				return o->TestBit(BIT(b));
			}

			//	for clarity - separate
			MonitorElement* create1D(DQMStore::IBooker&, MEInfo const&);
			MonitorElement* create2D(DQMStore::IBooker&, MEInfo const&);
			MonitorElement* create3D(DQMStore::IBooker&, MEInfo const&);
			MonitorElement* createProf(DQMStore::IBooker&, MEInfo const&);
			MonitorElement* createProf2D(DQMStore::IBooker&, MEInfo const&);

		private:
			//	a Map: MEname -> ME*
			typedef boost::ptr_map<std::string, MonitorElement> MEMap;
			MEMap											_meMap;
			//	a List of MEs to be Reset each Event and LS
			std::vector<std::string>						_namesResetEv;
			std::vector<std::string>						_namesResetLS;
			std::vector<std::string>						_namesToUpdate;
			//	Parameter Set of MEs	
			edm::ParameterSet const&						_ps;
			int												_debug;
			bool											_wasRetr;
			bool											_wasBooked;
			std::string										_subsystem;
	};

	/*
	 *	DEFs for Monitor Elements
	 */

}

#define GETPAR(PS, TYPE, NAME) \
	PS.getUntrackedParameter< TYPE >(NAME)

#endif




