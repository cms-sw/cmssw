#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<iterator>
#include<utility>
#include<algorithm>

typedef std::pair<unsigned long long ,std::string> Item;

void print(Item const & item) {
  std::cout << item.first << " " << item.second <<"\n";
}

int main() {

  std::string file("iovdump.txt");

  // parse iov dump for version 180
  std::ifstream in(file.c_str());

  std::string dummy;
  std::string tag;
  std::string contName;

  unsigned long long since, till;
  std::string token;
  std::vector<Item> values;

  in >> dummy >> tag;
  in >> dummy >> contName;
  char buff[1024];
  in.getline(buff,1024);
  in.getline(buff,1024);
  std::cout << buff << std::endl;
  char p;
  bool first=true;
  unsigned long long firstSince;
  while(in) {
    in.get(p); if (p=='T') break;
    in.putback(p);
    in >> since >> till >> token;  in.getline(buff,1024);
    values.push_back(Item(till,token));
    if (first) {
      first=false;
      firstSince=since;
    }
  }

  std::cout << tag << " " << contName 
	    << " " << firstSince << std::endl;
  std::for_each(values.begin(),values.end(),&print);
  std::cout << std::endl;
  return 0;

}
