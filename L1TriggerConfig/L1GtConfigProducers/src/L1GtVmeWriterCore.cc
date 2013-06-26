/**
 * \class L1GtVmeWriterCore
 *
 *
 * Description: core class to write the VME xml file.
 *
 * Implementation:
 *    Core class to write the VME xml file: L1GtVmeWriter is an EDM wrapper for this class.
 *    L1GtVmeWriterCore can also be used in L1 Trigger Supervisor framework,  with another
 *    wrapper - it must be therefore EDM-framework free.
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date: 2013/05/23 16:50:09 $
 * $Revision: 1.4 $
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVmeWriterCore.h"

// system include files
#include <iostream>
#include <sstream>
#include <bitset>

// user include files
//   base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlDefinitions.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"

// constructor
L1GtVmeWriterCore::L1GtVmeWriterCore(const std::string& outputDir,
        const std::string& vmeXmlFile) :
    m_outputDir(outputDir), m_vmeXmlFile(vmeXmlFile)
{

    object2reg_[Mu]=0x00A0000;
    object2reg_[NoIsoEG]=0x0020000;
    object2reg_[IsoEG]=0x0000000;
    object2reg_[ForJet]=0x0080000;
    object2reg_[TauJet]=0x0060000;
    object2reg_[CenJet]=0x0040000;
    object2reg_[HTT]=0x0100000;
    object2reg_[ETT]=0x0100000;
    object2reg_[ETM]=0x0100000;

    type2reg_[Type1s]=0x000C000;
    type2reg_[Type2s]=0x0004000;
    type2reg_[Type2wsc]=0x0008000;
    type2reg_[Type3s]=0x0010000;
    type2reg_[Type4s]=0x0000000;
    type2reg_[TypeETM]=0x0004000;
    type2reg_[TypeETT]=0x0000000;
    type2reg_[TypeHTT]=0x0008000;

    reg2hex_[m_xmlTagEtThreshold+"_1"]=0x0000000;
    reg2hex_[m_xmlTagEtThreshold+"_2"]=0x0000002;
    reg2hex_[m_xmlTagEtThreshold+"_3"]=0x0000004;
    reg2hex_[m_xmlTagEtThreshold+"_4"]=0x0000006;
    reg2hex_[m_xmlTagPtHighThreshold+"_1"]=0x0000000;
    reg2hex_[m_xmlTagPtHighThreshold+"_2"]=0x0000002;
    reg2hex_[m_xmlTagPtHighThreshold+"_3"]=0x0000004;
    reg2hex_[m_xmlTagPtHighThreshold+"_4"]=0x0000006;
    reg2hex_[m_xmlTagQuality+"_1"]=0x0000050;
    reg2hex_[m_xmlTagQuality+"_2"]=0x0000052;
    reg2hex_[m_xmlTagQuality+"_3"]=0x0000054;
    reg2hex_[m_xmlTagQuality+"_4"]=0x0000056;
    reg2hex_[m_xmlTagPtLowThreshold+"_1"]=0x0000058;
    reg2hex_[m_xmlTagPtLowThreshold+"_2"]=0x000005A;
    reg2hex_[m_xmlTagPtLowThreshold+"_3"]=0x000005C;
    reg2hex_[m_xmlTagPtLowThreshold+"_4"]=0x000005E;
    reg2hex_[m_xmlTagChargeCorrelation]=0x000008A;
    reg2hex_["jet_cnt_threshold"]=0x0000000;
    reg2hex_["threshold_lsb"]=0x0000000;
    reg2hex_["threshold_msb"]=0x0000002;

    // sepcial maps for jet counts
    jetType2reg_[0]=0x0000000;
    jetType2reg_[1]=0x0004000;
    jetType2reg_[2]=0x0008000;
    jetType2reg_[3]=0x000C000;
    jetType2reg_[4]=0x0010000;
    jetType2reg_[5]=0x0014000;

    jetObj2reg_[0]=0x00C0000;
    jetObj2reg_[1]=0x00E0000;
    
    spacesPerLevel_ = 2;

}

// destructor
L1GtVmeWriterCore::~L1GtVmeWriterCore()
{

    // empty now

}

int L1GtVmeWriterCore::condIndex2reg(const unsigned int &index)
{
    int indexcp = index;

    return (indexcp << 8);

}

std::string L1GtVmeWriterCore::calculateJetsAddress(const int &countIndex,
        const int &obj, const int &index)
{
    int result = jetType2reg_[countIndex] + jetObj2reg_[obj]
            + reg2hex_["jet_cnt_threshold"] +condIndex2reg(index);

    std::bitset<24> bs(result);

    std::ostringstream buffer;
    buffer<<bs;

    return buffer.str();
}

std::string L1GtVmeWriterCore::calculateAddress(const L1GtObject &obj,
        const L1GtConditionType &type, const std::string &reg, const int &index)
{

    int result = object2reg_[obj]+ type2reg_[type]+ reg2hex_[reg]
            +condIndex2reg(index);

    std::bitset<25> bs(result);

    std::ostringstream buffer;
    buffer<<bs;
    return buffer.str();

}

// destructor
std::string L1GtVmeWriterCore::openTag(const std::string &tag)
{
    return "<"+tag+">\n";
}

std::string L1GtVmeWriterCore::closeTag(const std::string &tag)
{
    return "</"+tag+">\n";
}

std::string L1GtVmeWriterCore::vmeAddrValueBlock(const std::string &addr,
        const int &val, const int &spaceLevel, const bool setMsb)
{
    std::ostringstream buffer;

    std::bitset<8> bsVal(val);

    if (setMsb)
        bsVal.set(7);

    buffer << spaces(spaceLevel) << openTag(m_xmlTagVmeAddress)
            << spaces(spaceLevel+1) << addr << std::endl
            << spaces(spaceLevel+1) << openTag(m_xmlTagValue)
            << spaces(spaceLevel+1) << bsVal <<std::endl
            << spaces(spaceLevel+1) << closeTag(m_xmlTagValue)
            << spaces(spaceLevel) << closeTag(m_xmlTagVmeAddress);

    return buffer.str();

}

std::string L1GtVmeWriterCore::spaces(const unsigned int &level)
{
    std::ostringstream buffer;

    for (unsigned int i=0; i<(level*spacesPerLevel_); i++)
    {
        buffer<<" ";
    }

    return buffer.str();

}

void L1GtVmeWriterCore::writeVME(const std::vector<ConditionMap>& _conditionMap,
        const std::map<std::string,int>& _cond2intMap, const L1GtVhdlTemplateFile& _header, const int spacesPerLevel)
{
    std::vector<ConditionMap> conditionMap = _conditionMap;
    std::map<std::string,int> cond2intMap = _cond2intMap;
    L1GtVhdlTemplateFile header = _header;
    L1GtVhdlDefinitions maps;
    
    // define appearance
    spacesPerLevel_ = spacesPerLevel;

    std::string fileName = m_outputDir+m_vmeXmlFile;

    // open output file
    std::ofstream outputFile(fileName.c_str());

    // begin of VME XML
    outputFile << "<" << m_xmlTagVme << ">" << std::endl;

    // insert header
    outputFile << spaces(1) << openTag(m_xmlTagHeader);
    header.substitute("vhdl_file_name", m_vmeXmlFile);
    header.removeEmptyLines();

    std::vector<std::string> lines = header.returnLines();

    for (unsigned int i = 0; i< lines.size(); i++)
    {
        outputFile << spaces(2)<<lines.at(i);
        //if (i!=lines.size()-1)
        outputFile<<std::endl;
    }

    outputFile << spaces(1) << closeTag(m_xmlTagHeader);

    // loop over chips
    for (unsigned int index=1; index<=2; index++)
    {
        outputFile << spaces(1) <<"<" << m_xmlTagChip << index <<">"
                << std::endl;

        // loop over condition map
        for (ConditionMap::iterator iterCond = conditionMap.at(index-1).begin(); iterCond != conditionMap.at(index-1).end(); iterCond++)
        {

            // open a condition
            outputFile << spaces(2) << "<" << iterCond->first << " "
                    << m_xmlConditionAttrObject <<"=\"" << maps.obj2str((iterCond->second->objectType())[0])<<"\" " <<m_xmlConditionAttrType <<"=\""
                    << maps.type2str(iterCond->second->condType())<< "\">"
                    <<std::endl;

            switch ((iterCond->second)->condCategory())
            {

            case CondMuon:
            {

                L1GtMuonTemplate* muonTemplate =
                        static_cast<L1GtMuonTemplate*>(iterCond->second);
                const std::vector<L1GtMuonTemplate::ObjectParameter>* op =
                        muonTemplate->objectParameter();

                // get the number of objects
                unsigned int nObjects = iterCond->second->nrObjects();

                for (unsigned int i =0; i<nObjects; i++)
                {

                    std::string opi = L1GtVhdlWriterCore::int2str(i);

                    // ptHighTreshold
                    std::string address = calculateAddress(Mu, (iterCond->second)->condType(),
                            (m_xmlTagPtHighThreshold+"_"+opi),
                            cond2intMap[iterCond->first]);

                    outputFile << spaces(3) << openTag(m_xmlTagPtHighThreshold)
                            <<vmeAddrValueBlock(address, (*op).at(i).ptHighThreshold, 4,iterCond->second->condGEq()) << spaces(3)
                            << closeTag(m_xmlTagPtHighThreshold);

                    // ptLow Threshold
                    address = calculateAddress(Mu, (iterCond->second)->condType(),
                            (m_xmlTagPtLowThreshold+"_"+opi),
                            cond2intMap[iterCond->first]);

                    outputFile << spaces(3) << openTag(m_xmlTagPtLowThreshold)
                            <<vmeAddrValueBlock(address, (*op).at(i).ptLowThreshold, 4, iterCond->second->condGEq()) << spaces(3)
                            <<closeTag(m_xmlTagPtLowThreshold);

                    // Quality
                    address = calculateAddress(Mu, (iterCond->second)->condType(), (m_xmlTagQuality+"_"
                            +opi), cond2intMap[iterCond->first]);

                    outputFile << spaces(3) << openTag(m_xmlTagQuality)
                            << vmeAddrValueBlock(address, (*op).at(i).qualityRange, 4) << spaces(3)
                            << closeTag(m_xmlTagQuality);

                }

                const L1GtMuonTemplate::CorrelationParameter *cp =
                        muonTemplate->correlationParameter();

                // Charage correlation -  occurs only one time
                outputFile << spaces(3)<< openTag(m_xmlTagChargeCorrelation);

                std::string address = calculateAddress(Mu, (iterCond->second)->condType(),
                        (m_xmlTagChargeCorrelation),
                        cond2intMap[iterCond->first]);

                outputFile << vmeAddrValueBlock(address, (*cp).chargeCorrelation, 4) << spaces(3)
                        << closeTag(m_xmlTagChargeCorrelation);

            }
                break;

            case CondCalo:
            {

                L1GtCaloTemplate* m_gtCaloTemplate =
                        static_cast<L1GtCaloTemplate*>(iterCond->second);
                const std::vector<L1GtCaloTemplate::ObjectParameter>* op =
                        m_gtCaloTemplate->objectParameter();

                // get the number of objects
                unsigned int nObjects = iterCond->second->nrObjects();

                for (unsigned int i =0; i<nObjects; i++)
                {

                    std::string opi = L1GtVhdlWriterCore::int2str(i);
                    std::string address = calculateAddress((iterCond->second->objectType()).at(0), (iterCond->second)->condType(),
                            (m_xmlTagPtHighThreshold+"_"+opi),
                            cond2intMap[iterCond->first]);

                    // insert Address/Value Tag
                    outputFile<<vmeAddrValueBlock(address, (*op).at(i).etThreshold, 3, iterCond->second->condGEq());

                }

            }
                break;

            case CondEnergySum:
            {

                L1GtEnergySumTemplate* energySumTempl =
                        static_cast<L1GtEnergySumTemplate*>(iterCond->second);

                const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op =
                        energySumTempl->objectParameter();

                // get the number of objects
                unsigned int nObjects = iterCond->second->nrObjects();

                for (unsigned int i =0; i<nObjects; i++)
                {

                    std::string opi = L1GtVhdlWriterCore::int2str(i);

                    std::string address = calculateAddress((iterCond->second->objectType()).at(0), (iterCond->second)->condType(),
                            (m_xmlTagPtHighThreshold+"_"+opi),
                            cond2intMap[iterCond->first]);

                    // insert Address/Value Tag
                    outputFile<<vmeAddrValueBlock(address, (*op).at(i).etThreshold, 3,iterCond->second->condGEq());

                }

            }

                break;

            case CondJetCounts:
            {

                L1GtJetCountsTemplate* jetsTemplate =
                        static_cast<L1GtJetCountsTemplate*>(iterCond->second);
                const std::vector<L1GtJetCountsTemplate::ObjectParameter>* op =
                        jetsTemplate->objectParameter();

                // get the number of objects
                // unsigned int nObjects = iterCond->second->nrObjects();

                // count index
                int ci = (*op)[0].countIndex;

                // 0 for count index 0-5 and 1 for count index 6 - 11
                int obj;

                if (ci<=5)
                    obj=0;
                else
                    obj=1;

                std::string address = calculateJetsAddress(ci, obj,
                        cond2intMap[iterCond->first]);

                outputFile<<vmeAddrValueBlock(address, (*op).at(0).countThreshold, 3, iterCond->second->condGEq());

            }
                break;
            case CondCorrelation:
            {
                // empty
            }
                break;

            case CondNull:
            {
                // empty
            }
                break;

            default:
            {
                // empty
            }
                break;

            }

            // close the condition
            outputFile << spaces(2)<< closeTag(iterCond->first);

        }

        outputFile << spaces(1) << "</" << m_xmlTagChip << index<<">"
                <<std::endl;

    }

    // end of vme XML
    outputFile <<closeTag(m_xmlTagVme);

    // close output file
    outputFile.close();

    std::cout << std::endl <<"*******   VME XML File: "<< m_outputDir
            << m_vmeXmlFile << " written sucessfully!  *******"<<std::endl;

}
