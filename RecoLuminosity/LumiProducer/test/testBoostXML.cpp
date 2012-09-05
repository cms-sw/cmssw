#include <iostream>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
struct Connection{
  std::string name;
  struct AuthParam{
      std::string user;
      std::string password;
  };
};

typedef std::vector<Connection> result;

void loadxml(const std::string &filename,
	     const std::string &conn,
	     std::string &user,
	     std::string &pass)
{
    using boost::property_tree::ptree;
    ptree pt;

    read_xml(filename, pt);

    BOOST_FOREACH(ptree::value_type const&v,pt.get_child("connectionlist")){
      if(v.first=="connection"){
	std::string con=v.second.get<std::string>("<xmlattr>.name",std::string(""));
	if(con==conn){
	  BOOST_FOREACH(ptree::value_type const&p,v.second){
	    if(p.first=="parameter"){
	      std::string name=p.second.get<std::string>("<xmlattr>.name");
	      std::string value=p.second.get<std::string>("<xmlattr>.value");
	      if(name=="user"){
		user=value;
	      }
	      if(name=="password"){
		pass=value;
	      }
	    }
	  }
	}
      }
    }
}
int main(){
  std::string conn("oracle://cms_orcon_adg/cms_lumi_prod");
  std::string user("");
  std::string pass("");
  loadxml("/afs/cern.ch/cms/lumi/DB/authentication.xml","oracle://cms_orcon_adg/cms_lumi_prod",user,pass);
  std::cout<<"user "<<user<<" , "<<pass<<std::endl;
}
