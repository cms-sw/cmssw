//$Id
/*
  A C++ commandline parser.
*/
/* ****************************************************************************
 *
 *  CommandLine.h
 *  Created by John F. Hubbard, on Fri Aug 11 2000, 16:04:24 PST
 *
 *  Copyright (c) 2000, ATD Azad Technology Development Corporation
 *
 *            The Reliable Software Outsource Resource
 *        You hire us, we do it for you, and we do it right.
 *
 *                       www.azadtech.com
 *
 *  Permission is granted to use this code without restriction,
 *  as long as this copyright notice appears in all source files.
 *
 *  File Contents: Interface and documentation of the CommandLine class.
 *
 * %version: 2.30 %
 * %date_modified: Thu Nov 02 16:10:20 2000 %
 * %created_by: jhubbard %
 *
 *************************************************************************** */

#ifndef COND_COMMANDLINE_H
#define COND_COMMANDLINE_H
#include <string>
#include <map>
#include <sstream>

namespace cond
{
      class CommandLine
    {
    public:
        typedef std::map<std::string, std::string> ARGMAPTYPE;
	CommandLine(int argc, char* argv[]) throw (std::string);
	CommandLine(int argc, char* argv[], const std::string& strFileName) throw (std::string);
 /** Command line arguments may also be supplied exclusively from a file.
 */
	CommandLine(const std::string& strFileName) throw (std::string);

        virtual ~CommandLine();
	std::string GetByName( const std::string& strArgName ) const throw (std::string);
	inline bool Exists( const std::string& strArgName ) const;
	inline int Count() const;
	virtual std::string Usage();
        void DumpDiagnostics() const;
	inline std::string GetProgramName() const;
    private:
        inline bool 		IsParamName(const std::string& strValue) const;
	inline std::string 	GetParamName(const std::string& strWord) const;
	void  ParseCommandLine(int argc, char* argv[])  throw (std::string);
	void  ParseSettings(const std::string& strInput) throw (std::string);
        std::string LoadSettingsFile(const std::string& strFileName) throw (std::string);
	void  TranslateQuotes(std::istringstream& ist, std::string& strWord);
	int     		mnParameterCount;
        ARGMAPTYPE  	mArgMap;
	std::string	mstrProgramName;
        // No copying or assignment is supported, at this point:
        CommandLine(const CommandLine&);
        CommandLine& operator=(const CommandLine&);
    };
      //inline functions
  inline int
    CommandLine::Count() const
    {
      return mnParameterCount;
    }
  
  inline bool
    CommandLine::IsParamName(const std::string& strWord) const
    {
      return ( (strWord.length()) > 1 && ( strWord[0] == '-' ) );
    }
  
  inline std::string
    CommandLine::GetParamName(const std::string& strWord) const
    {
      if (IsParamName(strWord) )
	{
	  return strWord.substr(1, strWord.length() - 1);
	}
      return strWord;
    }
    
  inline bool
    CommandLine::Exists( const std::string& strArgName ) const
    {
      return( mArgMap.find(strArgName) != mArgMap.end() );
    }
  
  inline std::string
    CommandLine::GetProgramName() const
    {
      return mstrProgramName;
    }
  
} //ns cond
#endif


