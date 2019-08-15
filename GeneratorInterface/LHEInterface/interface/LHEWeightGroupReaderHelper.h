#ifndef GeneratorInterface_LHEInterface_LHEWeightGroupReaderHelper_h
#define GeneratorInterface_LHEInterface_LHEWeightGroupReaderHelper_h

#include <string>
#include <vector>
#include <map>
#include <regex>
#include <fstream>

#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#include <tinyxml2.h>
using namespace tinyxml2;


class LHEWeightGroupReaderHelper {
public:
    LHEWeightGroupReaderHelper();

    //// possibly add more versions of this functions for different inputs
    void parseLHEFile(std::string filename);
    void parseWeightGroupsFromHeader(std::vector<std::string> lheHeader);


    edm::OwnVector<gen::WeightGroupInfo> getWeightGroups() {return weightGroups_;}
private:
    void loadAttributeNames(std::string baseName, std::vector<std::string> altNames ={});
    std::string toLowerCase(const char*);
    std::string toLowerCase(const std::string);
    std::map<std::string, std::string> getAttributeMap(std::string);
    std::string sanitizeText(std::string);
    bool isAWeight(std::string);
    
    // Variables
    edm::OwnVector<gen::WeightGroupInfo> weightGroups_;
    std::regex weightGroupStart_;
    std::regex weightGroupEnd_;
    std::regex weightContent_;
    
    std::map<std::string, std::string> nameConvMap;
};


std::string
LHEWeightGroupReaderHelper::toLowerCase(const char* name) {
    std::string returnStr;
    for (size_t i = 0; i < strlen(name); ++i)
        returnStr.push_back(tolower(name[i]));
    return returnStr;
}

std::string
LHEWeightGroupReaderHelper::toLowerCase(const std::string name) {
    std::string returnStr = name;
    transform(name.begin(), name.end(), returnStr.begin(), ::tolower);    
    return returnStr;

    
}

void LHEWeightGroupReaderHelper::loadAttributeNames(std::string baseName, std::vector<std::string> altNames) {
    for(auto altname : altNames) {
        nameConvMap[altname] = baseName;
    }
    nameConvMap[baseName] = baseName;
}

std::string
LHEWeightGroupReaderHelper::sanitizeText(std::string line) {
    std::map<std::string, std::string > replaceMap = {{"&lt;", "<"}, {"&gt;", ">"}};

    for(auto pair: replaceMap) {
	std::string badText = pair.first;
	std::string goodText = pair.second;
	while(line.find(badText) != std::string::npos) {
	    size_t spot = line.find(badText);
	    line.replace(spot, badText.size(), goodText);
	}
    }
    return line;
}


LHEWeightGroupReaderHelper::LHEWeightGroupReaderHelper() {
    weightGroupStart_ = std::regex(".*<weightgroup.+>.*\n*");
    weightGroupEnd_ = std::regex(".*</weightgroup>.*\n*");
    
    std::cout << "Init" << "\n";
    
    /// Might change later, order matters and code doesn't pick choices

    // Used for translating different naming convention to a common one
    loadAttributeNames("muf", {"facscfact"});
    loadAttributeNames("mur", {"renscfact"});
    loadAttributeNames("id");
    loadAttributeNames("pdf", {"pdf set", "lhapdf", "pdfset"});
    loadAttributeNames("dyn_scale");

    loadAttributeNames("combine");
    loadAttributeNames("name", {"type"});

}

std::map<std::string, std::string>
LHEWeightGroupReaderHelper::getAttributeMap(std::string line) {
    XMLDocument xmlParser;
    int error = xmlParser.Parse(line.c_str());
    if (error) {
        std::cout << "we have a problem!" << "\n";
	return std::map<std::string , std::string >();
	//do something....
    }

    std::map<std::string, std::string> attMap;
    XMLElement* element = xmlParser.FirstChildElement();
    
    for( const XMLAttribute* a = element->FirstAttribute(); a; a=a->Next()) {
        attMap[nameConvMap[toLowerCase(a->Name())]] = a->Value();
    }
    // get stuff from content of tag if it has anything.
    // always assume format is AAAAA=(  )BBBB    (  ) => optional space
    if (element->GetText() == nullptr) {
        return attMap;
    }
    // This adds "content: " to the beginning of the content. not sure if its a big deal or?
    std::string content = element->GetText();
    attMap["content"] = content;
    
    std::regex reg("(?:(\\S+)=\\s*(\\S+))");
    std::smatch m;
    while(std::regex_search(content, m, reg)) {
	std::string key = nameConvMap[toLowerCase(m.str(1))];
	if (attMap[key] != std::string()) {
	    if (m[2] != attMap[key]) {
		std::cout << m.str(2) << " vs " << attMap[key];
		// might do something if content and attribute don't match?
		// but need to be careful since some are purposefully different
		// eg dyn_scale is described in content but just given a number
	    }
	}
	else {
	    attMap[key] = m.str(2);
	}
	content = m.suffix().str();
    }
    return attMap;
    
}

bool
LHEWeightGroupReaderHelper::isAWeight(std::string line) {
    XMLDocument xmlParser;
    int error = xmlParser.Parse(line.c_str());
    if (error) {
	return false;
	//do something....
    }
    XMLElement* element = xmlParser.FirstChildElement();
    return element;
}

// void
// LHEWeightGroupReaderHelper::parseLHEFile(std::string filename) {
//     std::ifstream file;
//     file.open(filename);

//     std::string line;
//     std::smatch matches;
//     // TODO: Not sure the weight indexing is right here, this seems to more or less
//     // count the lines which isn't quite the goal. TOCHECK!
//     int index = 0;
//     while(getline(file, line)) {
//      if(std::regex_match(line, weightGroupStart_)) {
//          std::string name = getMap_testAll(line, {weightGroupInfo})["name"];
                
//          //TODO: Fine for now, but in general there should also be a check on the PDF weights,
//          // e.g., it could be an unknown weight
            
//          if(std::regex_match(name, scaleWeightMatch_)) {
//              weightGroups_.push_back(new gen::ScaleWeightGroupInfo(line));
//              std::cout << "scale weight" << "\n";
//          }


//          else
//              weightGroups_.push_back(new gen::PdfWeightGroupInfo(line));

//          /// file weights
//          while(getline(file, line) && !std::regex_match(line, weightGroupEnd_)) {
//              auto tagsMap = getMap_testAll(line, regexOptions);
                
//              std::regex_search(line, matches, weightContent_);
//              // TODO: Add proper check that it worked
//              std::string content = matches[1].str();

//              auto& group = weightGroups_.back();
//              if (group.weightType() == gen::kScaleWeights) {
//                  float muR = std::stof(tagsMap["mur"]);
//                  float muF = std::stof(tagsMap["muf"]);
//                  auto& scaleGroup = static_cast<gen::ScaleWeightGroupInfo&>(group);
//                  scaleGroup.addContainedId(index, tagsMap["id"], line, muR, muF);
//              }
//              else
//                  group.addContainedId(index, tagsMap["id"], line);

//              index++;                        
//          }
//      }
//     }
// }

void
LHEWeightGroupReaderHelper::parseWeightGroupsFromHeader(std::vector<std::string> lheHeader) {
    // TODO: Not sure the weight indexing is right here, this seems to more or less
    // count the lines which isn't quite the goal. TOCHECK!
    int index = 0;
    bool foundGroup = false;
    
    for (std::string headerLine : lheHeader) {
	std::cout << "Header line is:" << headerLine;
	headerLine = sanitizeText(headerLine);
        std::cout << "Header line is:" << weightGroups_.size() << " "<< headerLine;
	//TODO: Fine for now, but in general there should also be a check on the PDF weights,
        // e.g., it could be an unknown weight
        
        if (std::regex_match(headerLine, weightGroupStart_)) {
            //std::cout << "Adding new group for headerLine" << std::endl;
            foundGroup = true;
            std::string fullTag = headerLine + "</weightgroup>";
            auto groupMap = getAttributeMap(fullTag);
            std::string name = groupMap["name"];
                
            if(name.find("Central scale variation") != std::string::npos ||
               name.find("scale_variation") != std::string::npos) {
                weightGroups_.push_back(new gen::ScaleWeightGroupInfo(headerLine));
                std::cout << "scale weight" << "\n";
            }
	    else
		weightGroups_.push_back(new gen::PdfWeightGroupInfo(headerLine));
        }
        /// file weights
        else if (foundGroup && isAWeight(headerLine)) {
            //std::cout << "Adding new weight for headerLine" << std::endl;
            auto tagsMap = getAttributeMap(headerLine);
	    for(auto pair: tagsMap) {
		std::cout << pair.first << ": " << pair.second << " | ";
	    }
            std::cout << "\n";
            
            std::string content = tagsMap["content"];
            if (tagsMap["id"] == std::string()) {
                std::cout << "error" << "\n";
                // should do something
            }
            
            auto& group = weightGroups_.back();
            if (group.weightType() == gen::kScaleWeights) {
                if (tagsMap["mur"] == std::string() || tagsMap["muf"] == std::string()) {
                    std::cout << "error" << "\n";
                    // something should happen here
		    continue;
                }
                float muR = std::stof(tagsMap["mur"]);
                float muF = std::stof(tagsMap["muf"]);
                std::cout << tagsMap["id"] << " " << muR << " " << muF << " " << content << "\n";
                auto& scaleGroup = static_cast<gen::ScaleWeightGroupInfo&>(group);
		scaleGroup.addContainedId(index, tagsMap["id"], headerLine, muR, muF);
            }
            else
                group.addContainedId(index, tagsMap["id"], headerLine);
            index++;
	}
	// commented out since code doesn't work with this in....
	// else if(isAWeight(headerLine)) {
	//     // found header. Don't know what to do with it so just shove it into a new weightgroup for now
	//     // do minimum work for it
	//     weightGroups_.push_back(new gen::PdfWeightGroupInfo(headerLine));
	//     auto& group = weightGroups_.back();
	//     auto tagsMap = getAttributeMap(headerLine);
	//     group.addContainedId(index, tagsMap["id"], headerLine);
	//     foundGroup = true;
	//     index++;
	// }
	
	else if(std::regex_match(headerLine, weightGroupEnd_)) {
	    foundGroup = false;
	} 
	else {
            std::cout << "problem!!!" << "\n";
	}
    }
}



#endif


