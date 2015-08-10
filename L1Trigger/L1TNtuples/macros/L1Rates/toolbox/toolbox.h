#ifndef toolbox_h
#define toolbox_h

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

class toolbox{

public :
   
   toolbox();
   virtual ~toolbox();

   int getColor(int histIndex);

   char* toCStr(ostringstream &buf);

   void readConfigFile(std::string fileN, map<string,string> &content);

   std::vector<string> readFile(const std::string &fileName);

   std::string getMacroDir();

   template<typename T> static T convertFromString(const std::string& str, const unsigned int base=10)
   {
   	std::stringstream ss;
	T ret;

	switch(base)
	{
		case  8: ss<<std::oct; break;
		case 10: ss<<std::dec; break;
		case 16: ss<<std::hex; break;
		default: throw std::runtime_error("The \"base\" argument in toolbox::convertFromString must be either 8, 10 or 16.");
	}
	ss<<str;

	ss>>ret;
	if(ss.fail())
		throw std::runtime_error("Failed to convert from string to desired type.");

	return ret;
   }
};

#endif



#ifdef toolbox_cxx
toolbox::toolbox()
{

}

toolbox::~toolbox()
{
 
}


int toolbox::getColor(int histIndex)
{
	if (histIndex<8)
        {
		return histIndex;
	} else
	{
		while (histIndex>8)
		{
			histIndex = histIndex-8;		
		}
		
		return histIndex;
	}

	return histIndex;

}

char* toolbox::toCStr(ostringstream &buf)
{
	return const_cast<char*> (buf.str().c_str());
}

void toolbox::readConfigFile(std::string fileN, map<string,string> &content)
{

std::vector<string> file = readFile(fileN);

	cout << "Prescales:" << endl;

	for (unsigned int i = 0; i < file.size(); i++) {

		string str = file.at(i);

		std::vector<string> buffer;

		int pos = str.find("=");
		if (pos != -1) {
			string param, value;
			param = str.substr(0, pos);
			value = str.substr(pos + 1, str.length() - pos - 1);
			
			cout << param << ":" << value << endl;

			content[param] = value;

		}
	}


}


std::vector<string> toolbox::readFile(const std::string &fileName)
{
	static const char commentIndicator = '%';

	std::ifstream inputFile(fileName.c_str(), std::ios::in);

	std::vector<std::string> lines;

	if(!inputFile.good())
		throw std::runtime_error(std::string("Failed to open ")+fileName+" for reading.");

	while(inputFile.good())
	{
		std::string line;
		getline(inputFile, line);

		if(line.length()==0 || line[0]==commentIndicator)
			continue;

		if(line[line.length()-1]=='\r')
			line=line.substr(0, line.length()-1);
		lines.push_back(line);
	}

	if(inputFile.bad())
		throw std::runtime_error(std::string("Failed to read ")+fileName);

	return lines;
}


std::string toolbox::getMacroDir()
{
	const char* const macroDir=getenv("L1RATES_DIR");
	if(!macroDir)
		throw std::runtime_error("The environment variable L1RATES_DIR is not set.");
	const std::string str(macroDir);
	if(str[str.length()-1]=='/')
		return str;
	return str+"/";
}





#endif // #ifdef toolbox_cxx
