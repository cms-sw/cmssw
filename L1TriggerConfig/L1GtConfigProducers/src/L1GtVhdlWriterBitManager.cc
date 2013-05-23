/**
 * \class L1GtVhdlBitManager
 *
 *
 * \Description This class builds the LUTS for the GT firmware. Furthermore it is providing some helpers
 *  for basic bit operations in binary and hex format.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterBitManager.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"

#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"

// system include files
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

L1GtVhdlWriterBitManager::L1GtVhdlWriterBitManager()
{
	hex2binMap_["0"]="0000";
	hex2binMap_["1"]="0001";
	hex2binMap_["2"]="0010";
	hex2binMap_["3"]="0011";
	hex2binMap_["4"]="0100";
	hex2binMap_["5"]="0101";
	hex2binMap_["6"]="0110";
	hex2binMap_["7"]="0111";
	hex2binMap_["8"]="1000";
	hex2binMap_["9"]="1001";

	hex2binMap_["A"]="1010";
	hex2binMap_["B"]="1011";
	hex2binMap_["C"]="1100";
	hex2binMap_["D"]="1101";
	hex2binMap_["E"]="1110";
	hex2binMap_["F"]="1111";

	hex2binMap_["a"]="1010";
	hex2binMap_["b"]="1011";
	hex2binMap_["c"]="1100";
	hex2binMap_["d"]="1101";
	hex2binMap_["e"]="1110";
	hex2binMap_["f"]="1111";
}


std::string L1GtVhdlWriterBitManager::readMapInverse(const std::map<std::string,std::string>& map,std::string value)
{
	std::map<std::string,std::string>::const_iterator iter = map.begin();
	while (iter!=map.end())
	{
		if ((*iter).second == value)
			return (*iter).first;
		iter++;
	}
	return "";
}


std::string L1GtVhdlWriterBitManager::hex2bin(std::string hexString)
{

	std::string temp;
	for (unsigned int i=0; i<hexString.length(); i++)
	{
		std::string str;
		str=hexString[i];

		temp+=hex2binMap_[str];
	}

	return temp;
}


std::string L1GtVhdlWriterBitManager::bin2hex(std::string binString)
{

	std::string temp;
	for (unsigned int i=1; i<=binString.length(); i++)
	{
		//std::cout<<i%4<<std::endl;
		if(i%4==0)
		{
			//std::cout<<"I'm here!"<<std::endl;
			std::string str;
			str = binString.substr(i-4,4);
			//std::cout<<str<<std::cout<<std::endl;
			temp+=readMapInverse(hex2binMap_,str);
		}
	}

	return temp;
}


std::string L1GtVhdlWriterBitManager::mirror(unsigned int offset, std::string hexString, bool hexOutput)
{
	std::string temp,binString;

	char digit;
	bool hexInput=false;

	// check weather input hex or binary
	for(unsigned int i = 0; i<hexString.length(); i++)
	{
		if (hexString[i]!='0' || hexString[i]!='1' )
		{
			hexInput=true;
			break;
		}
	}

	if (hexInput)
		binString = hex2bin(hexString);
	else binString=hexString;

	unsigned int i=0, len=0;
	len = binString.length();

	if(offset > len)
		return binString;

	for(i = 0; i < (len - offset)/2; i++)
	{
		digit = binString[i + offset];

		binString[i + offset] = binString[len - 1 - i];
		binString[len - 1 - i] = digit;

	}

	if (hexOutput)
		return bin2hex(binString);
	else return binString;

}


std::string L1GtVhdlWriterBitManager::capitalLetters(std::string hexString)
{

	unsigned int i = 0;
	while(i<hexString.length())
	{
		if(hexString[i] == 'a') hexString[i] = 'A';
		else if(hexString[i] == 'b') hexString[i] = 'B';
		else if(hexString[i] == 'c') hexString[i] = 'C';
		else if(hexString[i] == 'd') hexString[i] = 'D';
		else if(hexString[i] == 'e') hexString[i] = 'E';
		else if(hexString[i] == 'f') hexString[i] = 'F';
		i++;
	}

	return hexString;
}


std::string L1GtVhdlWriterBitManager::shiftLeft(std::string hexString)
{
	std::string binString = hex2bin(hexString);

	binString.erase(0,1);
	binString+="0";

	return bin2hex(binString);
}


std::string L1GtVhdlWriterBitManager::buildEtaMuon(const std::vector<L1GtMuonTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter)
{
	std::ostringstream ossEta;

	ossEta
		<<counter
		<<" => X\"";

	for (unsigned int i =0; i<num; i++)
	{
		std::ostringstream ossEtaRange;
		ossEtaRange << std::hex<< std::setw(16)<<std::setfill('0')<<(*op).at(i).etaRange<<std::dec;

		/*
		ossEta
			<<mirror(32,ossEtaRange.str());
		*/
		
		ossEta << ossEtaRange.str();
		
		if (num>0 && i!=num-1) ossEta <<"_";

	}

	ossEta<<"\",\n";
	return ossEta.str();

}


std::string L1GtVhdlWriterBitManager::buildEtaCalo(const std::vector<L1GtCaloTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter)
{
	std::ostringstream ossEta;

	ossEta
		<<counter
		<<" => X\"";

	// loop over all relevant components of object parameters

	for (unsigned int i =0; i<num; i++)
	{
		std::ostringstream ossEtaRange;

		ossEtaRange << std::hex <<std::setw(4)<<std::setfill('0')<< (*op).at(i).etaRange<<std::dec;

		/*
		std::string tempstr=hex2bin(ossEtaRange.str());
		tempstr[0]='0';
		tempstr[15]='0';
		*/
		
		//ossEta<<std::setw(4)<<std::setfill('0')<<mirror(8,bin2hex(tempstr));

		ossEta << ossEtaRange.str();
		
		if (num>0 && i!=num-1) ossEta <<"_";

	}

	ossEta<<"\",\n";
	return ossEta.str();
}


std::string L1GtVhdlWriterBitManager::buildPhiCalo(const std::vector<L1GtCaloTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter)
{
	std::ostringstream ossPhi;

	ossPhi
		<<counter
		<<" => X\"";

	for (unsigned int i =0; i<num; i++)
	{
		ossPhi << std::hex<< std::setw(8)<<std::setfill('0')<<(*op).at(i).phiRange<<std::dec;

		if (num>0 && i!=num-1) ossPhi <<"_";
	}

	ossPhi<<"\",\n";
	return capitalLetters(ossPhi.str());
}


std::string L1GtVhdlWriterBitManager::buildPhiEnergySum(const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter)
{
	std::ostringstream ossPhi,count;

	if ((*op).at(0).phiRange0Word!=0)
	{
		ossPhi << "000000000000000000"<<std::hex<<(*op).at(0).phiRange1Word<<(*op).at(0).phiRange0Word<<std::dec;

		count<<counter;

		return (count.str()+" => X\""+capitalLetters(ossPhi.str())+"\",\n");

	} else
	return "";
}


std::string L1GtVhdlWriterBitManager::buildPhiMuon(const std::vector<L1GtMuonTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter,bool high)
{
	std::ostringstream ossPhi;

	ossPhi
		<<counter
		<<" => X\"";

	for (unsigned int i =0; i<num; i++)
	{
		if (high)
			ossPhi << std::hex<< std::setw(2)<<std::setfill('0')<<(*op).at(i).phiHigh<<std::dec;
		else
			ossPhi << std::hex<< std::setw(2)<<std::setfill('0')<<(*op).at(i).phiLow<<std::dec;

		if (num>0 && i!=num-1) ossPhi <<"_";
	}

	ossPhi<<"\",\n";
	return capitalLetters(ossPhi.str());

}


std::string L1GtVhdlWriterBitManager::buildDeltaEtaCalo(const L1GtCaloTemplate::CorrelationParameter* &cp,const unsigned int &counter)
{
	std::ostringstream deltaEtaRange, dEta,res;

	//std::cout<<"b:"<<.hex2bin("00383800")<<std::endl;

	deltaEtaRange << std::hex<< std::setw(4)<<std::setfill('0')<<(*cp).deltaEtaRange<<std::dec;

	// mirror deltaEtaRang and shift the mirrored value to the left by one bit;
	// add the original value to the calculated value
	dEta<<hex2bin(shiftLeft(mirror(0,deltaEtaRange.str())))<<hex2bin(deltaEtaRange.str());

	std::string result = capitalLetters(bin2hex(dEta.str()));

	res<<counter<<" => X\""<<result<<"\",\n";

	return res.str();

}


std::string L1GtVhdlWriterBitManager::buildDeltaEtaMuon(const L1GtMuonTemplate::CorrelationParameter* &cp,const unsigned int &counter)
{
	std::ostringstream deltaEtaRange, dEta;
	deltaEtaRange << std::hex<< std::setw(16)<<std::setfill('0')<<(*cp).deltaEtaRange<<std::dec;

	// mirror deltaEtaRang and shift the mirrored value to the left by one bit;
	// add the original value to the calculated value

	std::string result = capitalLetters((shiftLeft(mirror(0,deltaEtaRange.str()))+deltaEtaRange.str()));

	dEta<<counter<<" => X\""<<result<<"\",\n";

	return dEta.str();

}


std::string L1GtVhdlWriterBitManager::buildDeltaPhiCalo(const L1GtCaloTemplate::CorrelationParameter* &cp,const unsigned int &counter)
{
	std::ostringstream dPhi,deltaPhiRange, result;
	//std::cout<<.hex2bin("03E0000000000F81")<<std::endl;
	//std::cout<<.hex2bin("0080000000000200")<<std::endl;

	deltaPhiRange << std::hex<<std::setw(3)<<std::setfill('0')<<(*cp).deltaPhiRange<<std::dec;

	std::string binString = hex2bin(deltaPhiRange.str());

	//std::cout <<"========================" <<std::endl;

	std::string help2 = binString.substr(2,binString.length()-2);
	std::string help1 = help2.substr(1);
	help1 = mirror(0,bin2hex(help1),false);

	// here delta phi is built
	result<<help1;

	// might be wrong - has to be tested with more reference values!
	result<<help2.substr(0,8);
	result<<"0";

	result<<"00000000000000000000000000000";

	result<<help1;

	result<<binString.substr(2,binString.length()-2);

	dPhi<<counter<<" => X\""<<bin2hex(result.str())<<"\",\n";

	//std::cout<<result<<std::endl;

	/*
	 * Code from old GTS:
	  bm_dphi = bm_dphi.Get(2);
	  BitMngr help1, help2 = bm_dphi;
	  help2.SetAt(9, '\0');
	  help1 = help2.Get(1);
	  help1.Mirror();
	  BitMngr nuller;
	  nuller.InitBin("00 00 00 00 00 00 00 00 00 00 00 00 00 00 0");
	  BitMngr result;
	  result = result + help1 + help2 + nuller + help1 + bm_dphi;
	dphi = result.GetHex();
	*/

	return capitalLetters(dPhi.str());
}


std::string L1GtVhdlWriterBitManager::buildDeltaPhiMuon(const L1GtMuonTemplate::CorrelationParameter* &cp,const unsigned int &counter)
{
	std::ostringstream dPhi,deltaPhiRange0,deltaPhiRange1,temp,result ;
	std::string tempstr;

	deltaPhiRange0 << std::hex<< std::setw(16)<<std::setfill('0')<<(*cp).deltaPhiRange0Word<<std::dec;
	//deltaPhiRange1  /*<< std::hex<< std::setw(3)<<std::setfill('0')*/<<(*cp).deltaPhiRange1Word<<std::dec;

	//mirror deltaPhiRange, shift the mirrored value left and convert it to binary format
	temp<<hex2bin(shiftLeft(mirror(0,deltaPhiRange0.str())));
	tempstr=temp.str();
	// set last bit of tempstr to one;
	tempstr[tempstr.length()-1]='1';

	// build delta eta as stringstreamtau
	result
		<<tempstr
	// insert 16 ones
		<<hex2bin("FFFF")
		<<hex2bin(deltaPhiRange0.str());

	dPhi<<counter<<" => X\""<<bin2hex(result.str())<<"\",\n";

	return dPhi.str();

}
