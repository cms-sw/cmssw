#include<iostream>
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

using namespace std;

int main()

{

   char mybuf[20];

   cout<<"Filling the local buffer with 0xee"<<endl;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     mybuf[idx] = 0xee;
   }
   cout<<endl;

   cout<<"Making an Other Message"<<endl;
   OtherMessageBuilder  done_msg(mybuf, Header::DONE);
   cout << "Reading from Msg" << endl;
   cout << "Msg Code: "<< done_msg.code() << endl;
   cout << "Msg Size: "<< done_msg.size() << endl; 

   cout<<endl;
   cout<<"Local buffer contents with the Other Message (hex):"<<endl;
   cout<<hex;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     cout<<" "<<(0xff&((int)mybuf[idx]));
   }
   cout<<dec<<endl;
   cout<<endl;

   cout << "Building View on Same Message" << endl;
   OtherMessageView other_view(mybuf);
   cout << "Reading From  View" << endl;
   cout << "Msg Code From View: "<< other_view.code() << endl;
   cout << "Msg Size From View: "<< other_view.size() << endl;
 
   cout<<endl;
   cout<<"Making a second Other Message with body size of 12 and msg code 10"<<endl;
   OtherMessageBuilder  second_msg(mybuf, 10, 12);
   cout << "Reading from Msg" << endl;
   cout << "Msg Code: "<< second_msg.code() << endl;
   cout << "Msg Size: "<< second_msg.size() << endl; 

   cout<<"Setting the second message body to Hello World"<<endl;
   char *bodyPtr = (char *) second_msg.msgBody();
   strcpy(bodyPtr, "Hello World");
   
   cout<<endl;
   cout<<"Local buffer contents with the second Other Message (hex):"<<endl;
   cout<<hex;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     cout<<" "<<(0xff&((int)mybuf[idx]));
   }
   cout<<dec<<endl;
   cout<<endl;

   cout << "Building View on second Message" << endl;
   OtherMessageView second_view(mybuf);
   cout << "Reading From  View" << endl;
   cout << "Msg Code From View: "<< second_view.code() << endl;
   cout << "Msg Size From View: "<< second_view.size() << endl;
 
   cout << "Second Message body = " << second_view.msgBody() << endl;
}
