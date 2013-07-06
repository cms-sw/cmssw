#ifndef L1GtConfigProducers_L1GtVhdlTemplateFile_h
#define L1GtConfigProducers_L1GtVhdlTemplateFile_h

/**
 * \class L1GtVhdlTemplateFile
 *
 *
 * \Description The routines of this class provide all necessary features to deal with the VHDL
 *  templates for the firmware code of the GT
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author Philipp Wagner
 *
 * $Date$
 * $Revision$
 *
 */

// system include files

#include <map>
#include <string>
#include <vector>

class L1GtVhdlTemplateFile
{
	private:

		/// internal files additionally have entries in parameterMap_
		/// for "normal" files parameterMap is empty
		bool intern_;
		/// containing the content of the VHDL file
		std::vector<std::string> lines_;
		/// containing the header information of internal files
		std::map<std::string,std::string> parameterMap_;

	public:

		/// standard constructor
		L1GtVhdlTemplateFile();
		/// constructor with filename
		L1GtVhdlTemplateFile(const std::string &filename);
		/// copy constructor
		L1GtVhdlTemplateFile(const L1GtVhdlTemplateFile& rhs);
		/// destructor
		~L1GtVhdlTemplateFile();
		/// replaces searchString with replaceString at it's first occurance in string
		static const bool findAndReplaceString(std::string &paramString, const std::string &searchString, const std::string &replaceString);
		/// opens a template file. If the header information shall be parsed intern has to be set to true
		bool open(const std::string &fileName, bool internal=false);
		/// saves the content of the template file to a local file (the content of parameterMap_ will not be saved!)
		bool save(const std::string &fileName);
		bool close();
		/// replaces searchString with replaceString
		bool substitute(const std::string &searchString, const std::string &replaceString);
		/// replaces the whole line containing atLine and inserts content instead of it
		bool insert(const std::string &atLine, const std::vector<std::string>& content);
		/// replaces the whole line containing atLine with the content of file
		bool insert(const std::string atLine, const L1GtVhdlTemplateFile& file);
		/// prints the content of the VHDL File (only lines_)
		void print() const;
		/// prints the parameter map
		void printParameterMap() const;
		/// returns a string vector with the current content of the VHDL File
		std::vector<std::string> returnLines() const;
		/// returns parameter map
		std::map<std::string,std::string> returnParameterMap() const;
		/// returns a vector with all substitution parameters that are found in the template file
		std::vector<std::string> getSubstitutionParametersFromTemplate() const;
		/// finds all substitution parameters in str and collects them in the vector parameters.
		/// This routine is used by getSubstitutionParametersFromTemplate();
		bool extractParametersFromString(const std::string &str, std::vector<std::string> &parameters) const;
		/// adds a line at the end of the the file with the content of str
		void append(const std::string &str);
		/// adds the content of file at the end of (*this); the parameter map won't be changed
		void append(const L1GtVhdlTemplateFile& file);
		/// removes all lines that contain the str
		bool removeLineWithContent(const std::string &str);
		/// deletes all empty lines in a template file
		bool removeEmptyLines();
		/// checks weather a char is a blank
		bool isBlank(const char &chr) const;
		/// seperates a string at all blanks and saves the elements in result
		bool split(const std::string &param, std::vector<std::string> &result) const;
		/// extracts all conditions from a algorithm
		void getConditionsFromAlgo(std::string condString, std::vector<std::string> &result) const;
		/// returns a string with the content of vector lines
		std::string lines2String() const;
		/// returns a parameter of a internal template file
		std::string getInternalParameter(const std::string &indentifier);

};
#endif											  /*L1GtConfigProducers_L1GtVhdlTemplateFile_h*/
