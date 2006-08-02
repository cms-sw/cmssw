#include<iostream>
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

using namespace std;

int main()

{

   char mybuf[16];
   cout<<"Making an Other Message"<<endl;
   OtherMessageBuilder  done_msg(mybuf, Header::DONE);
   cout << "Reading from Msg" << endl;
   cout << "Msg Code: "<< done_msg.code() << endl;
   cout << "Msg Size: "<< done_msg.size() << endl; 


   cout << "Building View on Same Message" << endl;
   OtherMessageView other_view(mybuf);
   cout << "Reading From  View" << endl;
   cout << "Msg Code From View: "<< other_view.code() << endl;
   cout << "Msg Size From View: "<< other_view.size() << endl;
 

}
