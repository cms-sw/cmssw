// -*- C++ -*-
//
// Package:     HcalOnlineDb
// Class  :     HcalAssistant
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Thu Jul 16 11:39:22 CEST 2009
// $Id$
//


#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalAssistant.h"


HcalAssistant::HcalAssistant()
{
  addQuotes();
  srand(time(0));
}


HcalAssistant::~HcalAssistant()
{
}


int HcalAssistant::addQuotes(){
  quotes.push_back("Fear is the path to the Dark Side...");
  quotes.push_back("You don't know the power of the Dark Side...");
  quotes.push_back("You must learn the ways of the Force!");
  quotes.push_back("Where's the money, Lebowski?!");
  quotes.push_back("You see what happens when you find a stranger in the Alps!!!?");
  quotes.push_back("You hear this? This is the sound of inevitability. This is the sound of your death. Goodbye, mr. Anderson");
  quotes.push_back("Welcome to the desert of the Real");
  quotes.push_back("In Tyler we trust");
  quotes.push_back("How about a little snack?..Let's have a snack now, we can get friendly later");
  quotes.push_back("Is he human? Hey, no need for name calling!");
  quotes.push_back("Frankly, my dear, I don't give a damn");
  quotes.push_back("I've a feeling we're not in Kansas anymore");
  quotes.push_back("What we've got here is failure to communicate");
  quotes.push_back("I love the smell of napalm in the morning!");
  quotes.push_back("I see stupid people");
  quotes.push_back("Stella! Hey, Stella!");
  quotes.push_back("Houston, we have a problem");
  quotes.push_back("Mrs. Robinson, you're trying to seduce me. Aren't you?");
  quotes.push_back("I feel the need - the need for speed!");
  quotes.push_back("He's got emotional problems. What, beyond pacifism?");
}



std::string HcalAssistant::getRandomQuote(){
  int _quotes_array_size = quotes.size();
  int _num = 1e10;
  while(_num>=_quotes_array_size){
    _num = rand()%_quotes_array_size;
  }
  return quotes[_num];
}


std::string HcalAssistant::getUserName(void){
  struct passwd * _pwd = getpwuid(geteuid());
  std::string _name(_pwd->pw_name);
  return _name;
}
