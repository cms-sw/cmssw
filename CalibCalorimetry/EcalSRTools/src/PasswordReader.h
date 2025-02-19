#ifndef PASSWORDREADER_H
#define PASSWORDREADER_H

#include <string>

class PasswordReader{
public:
  /** Read data password from a local file. File can contain password for
   * several user. The format is one record by line. Line must start with the
   * user login followed by the password. Login and password can be separated by
   * space(s), tab(s), a colon or a slash.
   * @param fileName file with password list
   * @param login of user to look for
   * @param password [out] password read from the file
   * @throw cms::Exception if file cannot be read or if the password was not
   * found
   */
  void readPassword(const std::string& fileName,
                    const std::string& user,
                    std::string& password);


  /** Function to split a string into tokens. Usage:
   * <pre>
   * int pos = 0;
   * string tok;
   * const string s = .....;     //string to tokenize
   * const string delim = " \t"; //list of token delimiters
   * while((tok = tokenize(s, delim, pos))!=string::npos){
   *   .... code using tok ...
   * }
   * </pre>
   * @param s string to tokenize
   * @param list of delimiters
   * @param [in,out] pos current scan position in the string
   */
  std::string tokenize(const std::string& s,
                       const std::string& delim,
                       size_t& pos) const;

  /** Trims unwanted characters (e.g. spaces) at start and end of a string.
   * @param s [in,out] input string
   * @param chars set of characters to trim
   */
  std::string trim(const std::string& s, const std::string& chars) const;
};
  
#endif //PASSWORDREADER_H not defined
