# include "DQM/TrackerCommon/interface/CgiReader.h"

/************************************************/
// Read any form and return a multimap with the elements

void CgiReader::read_form(std::multimap<std::string, std::string> &form_info)
{
  cgicc::Cgicc reader(in);

  std::vector<cgicc::FormEntry> entries = reader.getElements();

  //  std::cout << "Trying to read a form of " << entries.size() << " elements!" << std::endl;

  form_info.clear();

  for (unsigned int i = 0; i < entries.size(); i++)
    {
      std::string name  = entries[i].getName();
      std::string value = entries[i].getValue();

      //      std::cout << "Read " << name << " = " << value << std::endl;

      std::pair<std::string, std::string> map_entry(name, value);
      form_info.insert(map_entry);
    }
}

std::string CgiReader::read_cookie(std::string name)
{
  cgicc::Cgicc reader(in);
  std::string value;

  const cgicc::CgiEnvironment& env = reader.getEnvironment();

  cgicc::const_cookie_iterator it;
  for (it = env.getCookieList().begin(); 
       it != env.getCookieList().end();
       it ++)
    {
      if (name == it->getName())
	{
	  value = it->getValue();
	}
    }
  return value;
}

