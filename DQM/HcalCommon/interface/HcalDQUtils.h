#ifndef HCALDQUTILS_H
#define HCALDQUTILS_H

/*	
 *	file:				HcalDigiTask.h
 *	Author:				Viktor Khristenko
 *	Start Date:			03/04/2015
 *
 *	TODO:
 *		1) Other utilities???	
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

#include <vector>
#include <string>
#include "boost/ptr_container/ptr_map.hpp"
#include "boost/container/map.hpp"

namespace hcaldqm
{

	//	a struct of labels
	struct Labels
	{
		Labels(edm::ParameterSet const&);		
		edm::InputTag& operator[](std::string s) 
		{
			return _labels[s];
		}

		typedef boost::container::map<std::string, edm::InputTag> LabelMap;
		LabelMap _labels;
	};

	/*
	 *	Namespace for converting between different coordinate spaces
	 *	I call it packaging. Employes various classes
	 */
	namespace packaging
	{
		//	Resolve if this Trigger Tower comes from HF or HBHE
		bool isHFTrigTower(int absieta);
		bool isHBHETrigTower(int absieta);

		/*
		 *	Packager struct to package/unupackage coordinates
		 */
		struct Packager
		{
			Packager() {}
			Packager(int iphimin, int iphimax, int iphistep,
					int ietamin, int ietamax, int dmin, int dmax) :
				_iphimin(iphimin), _iphimax(iphimax), _iphistep(iphistep),
				_ietamin(ietamin), _ietamax(ietamax),
				_dmin(dmin), _dmax(dmax)
			{
				//	ieta step is the same for all subsystems
				_ietastep = hcaldqm::constants::STD_HF_STEPIETA;
				_ietanum = 2*((_ietamax-_ietamin)+1);
				_iphinum = (_iphimax-_iphimin)/_iphistep+1;
				_dstep = hcaldqm::constants::STD_HF_STEPDEPTH;
				_dnum = (_dmax-_dmin)/_dstep+1;
			}
			~Packager(){}
	
			int ieta(int iieta)
			{
				if (iieta<(_ietanum/2))
					return -(_ietamin+_ietastep*iieta);
				else
					return _ietamin + _ietastep*(iieta-_ietanum/2);
			}
	
			int iieta(int ieta)
			{
				if (ieta<0)
					return (abs(ieta)-_ietamin)/_ietastep;
				else
					return (ieta-_ietamin)/_ietastep+_ietanum/2;
			}

			int iphi(int iiphi)
			{
				return _iphimin+_iphistep*iiphi;
			}

			int iiphi(int iphi)
			{
				return (iphi - _iphimin)/_iphistep;
			}

			int depth(int id)
			{
				return _dmin+id*_dstep;
			}

			int idepth(int d)
			{
				return (d-_dmin)/_dstep;
			}

			int _iphimin, _iphimax, _iphinum, _iphistep;
			int _ietamin, _ietamax, _ietanum, _ietastep;
			int _dmin, _dmax, _dnum, _dstep;
		};
	}

	/*
	 *	Namespace containing all the Math Stuff.
	 */
	namespace math
	{
		//	Copmute a TS with max nominal fC
		template<typename Hit>
		int maxTS(Hit const& hit, double ped=0)
		{	
			int maxT = -1;
			double maxQ = -1;
			for (int i=0; i<hit.size(); i++)
				if ((hit.sample(i).nominal_fC()-ped)>maxQ)
				{
					maxQ = hit.sample(i).nominal_fC()-ped;
					maxT=i;
				}

			return maxT;
		}

		template<typename intdouble>
		int max(intdouble *arr, int i=0, int j=10)
		{
			double max=-1;
			int imax=-1;

			for (int ii=i; ii<=j; ii++)
				if (max<arr[ii])
				{
					imax = ii;
					max = arr[ii];
				}

			return imax;
		}

		template<typename intdouble>
		std::pair<double, double> meanrms(intdouble *arr, int i=0, int j=10)
		{
			double sum(0), sum2(0), sumn(0);
			double mean(-1), rms(-1);

			for (int ii=i; ii<=j; ii++)
			{
				sum += arr[ii]*ii;
				sumn += arr[ii];
			}
			if (sumn>0) 
				mean = sum/sumn;
			else
				return std::pair<double, double>(mean, rms);

			for (int ii=i; ii<=j; ii++)
				sum2+=arr[ii]*(ii-mean)*(ii-mean);
			rms = sqrt(sum2/sumn);
			return std::pair<double, double>(mean, rms);
		}

		//	Compute the nominal_fC-weighted Time average
		template<typename Hit>
		double aveT(Hit const& hit, double ped=0)
		{
			double sumQ = 0;
			double sumQT = 0;
			double ave = 0;
			for (int i=0; i<hit.size(); i++)
			{
				sumQ+=hit.sample(i).nominal_fC()-ped;
				sumQT+=(i+1)*(hit.sample(i).nominal_fC()-ped);
			}
			
			ave = sumQT/sumQ - 1;
			return ave;
		}

		//	Compute the Sum over the digi
		template<typename Hit>
		double sum(Hit const& hit, int i, int j, double ped=0, 
			bool useADC=false)
		{
			if (i<0 && j<hit.size())
				i=0;
			else if (i>=0 && j>=hit.size())
				j=hit.size()-1;

			double sumQ = 0;
			for (int ii=i; ii<=j; ii++)
				if (useADC)
					sumQ+=hit.sample(ii).adc()-ped;
				else
					sumQ+=hit.sample(ii).nominal_fC() - ped;

			return sumQ;
		}
	}

	/*
	 *	classes for Pedestal/Laser/LED Data Management
	 *	I found it's useful to pretty much leave as is in Dima's code
	 *	Classes:
	 *	-> HcalDQData - parent class for all of them.
	 *		Defines abstract methods
	 */
	class HcalDQData
	{
		public:
			HcalDQData() :
				_status(0),
				_mean(-1), _rms(-1), _refmean(-1), _refrms(-1),	
				_overflow(0),
				_underflow(0),
				_isRef(false),
				_entries(0)
			{}
			virtual ~HcalDQData() {}

			virtual void reset()
			{
				_mean = -1;
				_rms = -1;
				_refmean = -1;
				_rms = -1;
				_overflow = 0;
				_underflow = 0;
				_entries = 0;
				_status = 0;
			};
			virtual void setRef(double mean, double rms) 
			{
				_refmean=mean; 
				_refrms=rms; 
			}
			virtual void changeStatus(int s) {_status|=s;}
			virtual int getStatus() {return _status;}
			virtual std::pair<double, double> average() = 0;
			virtual int getEntries() {return _entries;}
			virtual void setEntries(int n) {_entries=n;}
			virtual int getOverflow() {return _overflow;}
			virtual int getUnderflow() {return _underflow;}
	
		protected:
			int			_status;
			double		_mean;
			double		_rms;
			double		_refmean;
			double		_refrms;
			int			_overflow;
			int			_underflow;
			bool		_isRef;
			int			_entries;
	};

	/*
	 *	Data for LED per Channel
	 */
	class HcalDQLedData : public HcalDQData
	{
		public:
			HcalDQLedData() {}
			virtual ~HcalDQLedData() {}

			virtual void reset()
			{
				HcalDQData::reset();
				_sumAmp=0;
				_sumAmp2=0;
				_sumTime=0;
				_sumTime2=0;
				_meanTime=-1;
				_rmsTime=-1;
			}

			template<typename Hit>
			void push(Hit const& hit, double ped=0,
					int tstogo=1)
			{
				double aveT = hcaldqm::math::aveT(hit, ped);
				double sumQ = hcaldqm::math::sum(hit, 
						hcaldqm::math::maxTS(hit, ped)-tstogo, 
						hcaldqm::math::maxTS(hit, ped)+tstogo, ped);

				if (sumQ<hcaldqm::constants::STD_MINLEDQ)
					_underflow++;
				else if (sumQ>hcaldqm::constants::STD_MAXLEDQ)
					_overflow++;
				else 
				{
					_entries++;
					_sumAmp		+= sumQ;
					_sumAmp2	+= sumQ*sumQ;
					_sumTime	+= aveT;
					_sumTime2	+= aveT*aveT;
				}
			}

			virtual std::pair<double, double> average()
			{
				if (_entries<=0)
					return std::pair<double, double>(_mean, _rms);

				_mean = _sumAmp/_entries;
				_rms = sqrt(_sumAmp2/_entries - _mean*_mean);
				return std::pair<double, double>(_mean, _rms);
			}

			virtual std::pair<double, double> averageTiming()
			{
				if (_entries<=0)
					return std::pair<double, double>(_meanTime, _rmsTime);

				_meanTime = _sumTime/_entries;
				_rmsTime = sqrt(_sumTime2/_entries - _meanTime*_meanTime);
				return std::pair<double, double>(_meanTime, _rmsTime);	
			}

		protected:
			double		_sumAmp;
			double		_sumAmp2;
			double		_sumTime;
			double		_sumTime2;
			double		_meanTime;
			double		_rmsTime;
	};

	/*
	 *	Data For Ped per channel
	 */
	class HcalDQPedData : public HcalDQData
	{
		public:
			HcalDQPedData() {}
			virtual ~HcalDQPedData(){}

			virtual void reset()
			{
				HcalDQData::reset();
				for (int i=0; i<hcaldqm::constants::STD_NUMADC; i++)
					_adc[i]=0;
			}

			virtual void push(unsigned int v)
			{
				if (v>=hcaldqm::constants::STD_NUMADC)
					_overflow++;
				else
				{
					_adc[v&0x7F]++;
					_entries++;
				}
			}

			virtual std::pair<double, double> average()
			{
				int imax = hcaldqm::math::max<int>(_adc, 0, 
						hcaldqm::constants::STD_NUMADC-1);	

				return std::pair<double, double>(
						hcaldqm::math::meanrms<int>(_adc, 0, 
							imax+hcaldqm::constants::STD_NUMBINSFORPED));
			}

		protected:
			int _adc[hcaldqm::constants::STD_NUMADC];
	};
}

#endif








