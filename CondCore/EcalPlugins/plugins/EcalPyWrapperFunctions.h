#ifndef CondCore_EcalPlugins_plugins_EcalPyWrapperFunctions_H
#define CondCore_EcalPlugins_plugins_EcalPyWrapperFunctions_H

#include <sstream>

//inherit this class, override getValues() function
//virtual class EcalPyWrapperStrategy {
//
//};
//class EcalMeanStrategy: EcalPyWrapperStrategy{
//	
//};
//class EcalMeanStrategy: EcalPyWrapperStrategy{
//	
//};

template <class T>
class EcalPyWrapperHelper{
public:
    static const unsigned int MEAN    = 0;
    static const unsigned int STATUS  = 1;

	EcalPyWrapperHelper(unsigned int totalValues, unsigned int status = 0, std::string names = "-Means: "):total_values(totalValues), status(status), names(names){}
	//~EcalPyWrapperHelper();
	std::string printBarrelsEndcaps( const std::vector<T> & barrelItems, const std::vector<T> & endcapItems) {
		std::stringstream ss;

		//print barrels:
		type_vValues barrelsVec = getValues(barrelItems);
		unsigned int barrelsTotal = barrelItems.size();
		ss << std::endl << "---Barrels. Total: " << barrelsTotal << std::endl;
			switch (status){
				case (MEAN)  : ss << names << std::endl; break;
				case (STATUS): ss << "-With errors: " << std::endl; break;// << ecalcond::bad(barrelItems) << std::endl; break;
				default	     : break;
			}
		ss << printValues(barrelsVec, barrelsTotal);

		//print endcaps:
		type_vValues endcapVec = getValues(endcapItems);
		unsigned int endcapTotal = endcapItems.size();
		ss << std::endl << "---Endcaps. Total: " << endcapTotal << std::endl;
			switch (status){
				case (MEAN)  : ss << names << std::endl; break;
				case (STATUS): ss << "-With errors: " << std::endl; break;// << ecalcond::bad(endcapItems) << std::endl; break;
				default	     : break;
			}
		ss << printValues(endcapVec, endcapTotal);	

		return ss.str();
	}
protected:
	unsigned int total_values;
	unsigned int status;
    std::string names; 
	typedef std::vector<std::pair< std::string, float> > type_vValues;

	//this needs to be overriden in inherited class:
	virtual type_vValues getValues( const std::vector<T> & vItems) = 0;
	/*EXAMPLE:
	class EcalPedestalsHelper: public EcalPyWrapperHelper<EcalPedestal>{
	public:
		type_vValues getValues( const std::vector<EcalPedestal> & vItems)
		{
			unsigned int totalValues = 6; 
			type_vValues vValues(totalValues);

			vValues[0].first = "mean_x12";
			vValues[1].first = "rms_x12";
			vValues[2].first = "mean_x6";
			vValues[3].first = "rms_x6";
			vValues[4].first = "mean_x1";
			vValues[5].first = "rms_x1";
			
			vValues[0].second = .0;
			vValues[1].second = .0;
			vValues[2].second = .0;
			vValues[3].second = .0;
			vValues[4].second = .0;
			vValues[5].second = .0;
			
			//get info:
			for(std::vector<EcalPedestal>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems){
				vValues[0].second += iItems->mean(1);
				vValues[1].second += iItems->rms(1);
				vValues[2].second += iItems->mean(2);
				vValues[3].second += iItems->rms(2);
				vValues[4].second += iItems->mean(3);
				vValues[5].second += iItems->rms(3);
			}
			return vValues;
		}
		*/
private:
	std::string printValues(const type_vValues & vValues, const unsigned int & total) {
		std::stringstream ss;

		for (type_vValues::const_iterator iVal = vValues.begin(); iVal != vValues.end(); ++iVal){
			switch (status){
				case (MEAN)  : ss << iVal->first << ": " << ((iVal->second)/total) << std::endl; break;
				case (STATUS): if (iVal->second != 0) {ss << iVal->first << ": " << ((iVal->second)) << std::endl; break;} else {break; }
				default	     : break;
			}
			
		}
		return ss.str();
	}

};
#endif