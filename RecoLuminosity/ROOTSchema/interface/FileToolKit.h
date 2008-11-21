#ifndef __FILETOOLKIT_H__
#define __FILETOOLKIT_H__

#include <string>
#include <vector>

class FileToolKit{
 public:

  FileToolKit(){}
  ~FileToolKit(){}
  
  // HTML specific
  void MakeEmptyWebPage(const std::string &fileName,
			const std::string &title);

  // File specific
  void InsertLineAfter( const std::string &fileName,
			const std::string &newLine,
			const std::string &searchLine);

  void InsertLineBefore( const std::string &fileName,
			 const std::string &newLine,
			 const std::string &searchLine);

  bool fileExists( const std::string &fileName );

  // String 

  void Tokenize(const std::string& str,
		std::vector< std::string >& tokens,
		const std::string& delimiters = " ");
  
  // File system specific
  
  int MakeDir( std::string dirName, mode_t writeMode );
    
};


#endif
