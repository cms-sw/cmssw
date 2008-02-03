/**
 * \class L1GtVhdlTemplateFile
 *
 *
 * Description: a class to deal with VHDL template files
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 * $Date: 2008/01/31 15:27:17 $
 * $Revision: 1.1 $
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"

// system include files
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

// constructor(s)

//standard constructor for a empty file
L1GtVhdlTemplateFile::L1GtVhdlTemplateFile()
{
    intern_=false;
}


//constructor which already loads a file
L1GtVhdlTemplateFile::L1GtVhdlTemplateFile(const std::string &filename)
{
    if (!open(filename,false)) std::cout<<"Error while opening file: "<<filename<<std::endl;
}


//copy constructor
L1GtVhdlTemplateFile::L1GtVhdlTemplateFile(const L1GtVhdlTemplateFile& rhs)
{
    lines_=rhs.lines_;
    intern_=rhs.intern_;
    parameterMap_=rhs.parameterMap_;

}


// destructor
L1GtVhdlTemplateFile::~L1GtVhdlTemplateFile()
{
    // empty
}


bool L1GtVhdlTemplateFile::findAndReplaceString(std::string &paramString, const std::string &searchString, const std::string &replaceString)
{
    size_t position;
    position = paramString.find(searchString);
    if (position == std::string::npos) return false;
    paramString.replace(position,searchString.length(),replaceString);
    return true;
}


bool L1GtVhdlTemplateFile::open(const std::string &fileName, bool internal)
{

    char buffer[2000];
    std::string stringBuffer;

    std::fstream inputFile(fileName.c_str(),std::ios::in);
    //check weather file has been opened successfully
    if(!inputFile.is_open()) return false;

    //store content of the template in Vector lines
    while(!inputFile.eof())
    {
        inputFile.getline(buffer,2000);
        stringBuffer=buffer;
        //Remove DOS seperators (For example if the template file was created under NT)
        if (stringBuffer[stringBuffer.length()-1]==13)
        {
            stringBuffer.replace(stringBuffer.length()-1,1,"");
        }
        //the current buffer + a seperator to the vector lines
        lines_.push_back(stringBuffer/*+"\n"*/);
    }

    inputFile.close();

    if (internal)
    {

        //Delete lines containing parameters after moving them to parameterMap_
        std::vector<std::string>::iterator iter = lines_.begin();
        while( iter != lines_.end() )
        {
            if ((*iter)[0]=='#')
            {
                std::vector<std::string>::iterator iter2 = iter;
                iter2++;
                parameterMap_[(*iter).substr(1)]=(*iter2);
                lines_.erase(iter);
                //iter is pointing on the line after the previously deleted line
                //which is the content of the parameter
                lines_.erase(iter);
            }
            iter++;
        }

        //remove empty lines
        iter = lines_.begin();
        while( iter != lines_.end() )
        {
            if ((*iter)=="" || (*iter).length()==0 || (*iter)=="    ") lines_.erase(iter); else
                iter++;
        }

    }

    return true;

}


bool L1GtVhdlTemplateFile::save(const std::string &fileName)
{
    std::ofstream outputFile(fileName.c_str());
    std::vector<std::string>::iterator iter = lines_.begin();

    //Write content of lines_ into the outputfile.
    while( iter != lines_.end() )
    {
        //std::cout<<"Last sign: "<<*iter[(*iter).length()-3];
        outputFile << *iter<<std::endl;
        iter++;
    }

    outputFile.close();
   
    return true;

}


bool L1GtVhdlTemplateFile::substitute(const std::string &searchString, const std::string &replaceString)
{

    bool success = false;

    std::vector<std::string>::iterator iter = lines_.begin();
    while( iter != lines_.end())
    {
        //The substitution parameter always appears as follows: $(parameter)
        while (findAndReplaceString(*iter,("$("+searchString+")"), replaceString))
        {
            findAndReplaceString(*iter,("$("+searchString+")"), replaceString);
            success = true;
        }
        iter++;
    }

    return success;

}


bool L1GtVhdlTemplateFile::insert(const std::string &atLine, std::vector<std::string> content)
{
    bool success = false;
    std::vector<std::string>::iterator iter = lines_.begin();

    //Loop until the substitution parameter is discovered the first time
    while( iter != lines_.end() )
    {
        //check, weather the current line is containing the substitution parameter
        if ((*iter).find(atLine)!=std::string::npos)
        {
            //Delete the line with the subsitution parameter
            iter = lines_.erase(iter);
            //insert the content of file
            lines_.insert(iter,content.begin(),content.end());

            success=true;
            break;
        }

        iter++;
    }

    return success;
}


bool L1GtVhdlTemplateFile::insert(const std::string atLine, L1GtVhdlTemplateFile file)
{
    std::vector<std::string>::iterator iter = lines_.begin();
    std::vector<std::string> temp = file.returnLines();

    if (insert(atLine,temp)) return true;

    return false;
}


bool L1GtVhdlTemplateFile::close()
{
    //empty
    return true;
}


void L1GtVhdlTemplateFile::print()
{
    std::vector<std::string>::iterator iter = lines_.begin();
    while( iter != lines_.end())
    {
        std::cout<<*iter<<std::endl;
        iter++;
    }

}


std::vector<std::string> L1GtVhdlTemplateFile::returnLines()
{
    return lines_;
}


void L1GtVhdlTemplateFile::printParameterMap()
{
    std::cout<<"Enter parametermap"<<std::endl;

    std::map<std::string,std::string>::iterator iter =  parameterMap_.begin();

    while( iter != parameterMap_.end())
    {
        std::cout<<(*iter).first<<": "<<(*iter).second<<std::endl;
        iter++;;
    }
}


std::map<std::string,std::string> L1GtVhdlTemplateFile::returnParameterMap()
{
    return parameterMap_;
}


bool L1GtVhdlTemplateFile::extractParametersFromString(const std::string &str, std::vector<std::string> &parameters)
{
    // check, weather the current line is containing a substitution parameter
    // the routine is making sure, that it's not extracting a parameter from
    // a comment
    if (int pos1=str.find("$(")!=std::string::npos && str.substr(0,2)!="--")
    {
        int pos2=str.find(")");
        // get the substituion parameter
        std::string tempStr=(str.substr(pos1+1,(pos2-pos1-1)));
        // return a pair with the substitution parameter and the
        // the rest of the string after the substitution parameter

        // here a should be checked, weather the vector is already containing
        // the parameter befor adding it.

        parameters.push_back(tempStr);
        //recursive call
        while (extractParametersFromString(str.substr(pos2), parameters)) extractParametersFromString(str.substr(pos2), parameters);

        return true;
    }
    else
    {
        return false;
    }
    
    return true;
}


std::vector<std::string> L1GtVhdlTemplateFile::getSubstitutionParametersFromTemplate()
{
    std::vector<std::string> temp;
    std::vector<std::string>::iterator iter = lines_.begin();

    // loop until the substitution parameter is discovered the first time
    while( iter != lines_.end() )
    {
        extractParametersFromString((*iter), temp);
        iter++;
    }

    return temp;

}


void L1GtVhdlTemplateFile::append(const std::string &str)
{
    lines_.push_back(str);
}


void L1GtVhdlTemplateFile::append(const L1GtVhdlTemplateFile& file)
{
    for (unsigned int i=0; i<file.lines_.size(); i++)
    {
        lines_.push_back(file.lines_.at(i));
    }
}


bool L1GtVhdlTemplateFile::removeLineWithContent(const std::string &str)
{
    bool success = false;

    std::vector<std::string>::iterator iter = lines_.begin();
    while( iter != lines_.end())
    {
        size_t position;
        position = (*iter).find(str);

        if (position != std::string::npos)
        {
            lines_.erase(iter);
            success=true;
        } else iter++;
    }
    return success;
}


bool L1GtVhdlTemplateFile::removeEmptyLines()
{
    std::vector<std::string>::iterator iter = lines_.begin();

    while( iter != lines_.end() )
    {
        if ((*iter)=="" || (*iter).length()==0 || (*iter)=="    ") lines_.erase(iter); else
            iter++;
    }

    return true;
}


bool L1GtVhdlTemplateFile::isBlank(const char &chr)
{
    if (chr==' ') return true;
    return false;

}


bool L1GtVhdlTemplateFile::split(const std::string &param, std::vector<std::string> &result)
{
    unsigned int i = 0;
    while (isBlank(param[i]))
    {
        i++;
    }

    std::string temp = param.substr(i);
    std::size_t pos = temp.find(" ");

    if (pos != std::string::npos)
    {
        std::string temp2 = temp.substr(0, pos);
        result.push_back(temp2);
        while (split(temp.substr(pos),result)) split(temp.substr(pos),result);

    } else if (!isBlank(temp[pos+1]))
    {
        result.push_back(temp);
        return false;
    } else
    return false;

    return false;
}


void L1GtVhdlTemplateFile::getConditionsFromAlgo(std::string condString, std::vector<std::string> &result)
{
    std::vector<std::string> operators;

    operators.push_back("AND");
    operators.push_back("OR");
    operators.push_back("NOT");
    operators.push_back("(");
    operators.push_back(")");

    for (unsigned int i =0; i<operators.size(); i++)
    {
        while (findAndReplaceString(condString, operators.at(i), "")) findAndReplaceString(condString, operators.at(i), "");
    }

    split(condString,result);

}


std::string L1GtVhdlTemplateFile::lines2String()
{
    std::vector<std::string>::iterator iter = lines_.begin();
    std::ostringstream buffer;

    while( iter != lines_.end() )
    {
        buffer<<(*iter)<<std::endl;
        iter++;

    }

    return buffer.str();
}


std::vector<std::string> L1GtVhdlTemplateFile::returnLinesVec()
{
    return lines_;
}
