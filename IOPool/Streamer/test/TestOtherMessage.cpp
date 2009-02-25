#include <iostream>
#include "IOPool/Streamer/interface/OtherMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include <cstring>

int main()

{

   unsigned char mybuf[20];

   std::cout << "Filling the local buffer with 0xee" << std::endl;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     mybuf[idx] = 0xee;
   }
   std::cout << std::endl;

   std::cout << "Making an Other Message" << std::endl;
   OtherMessageBuilder  done_msg(mybuf, Header::DONE);
   std::cout << "Reading from Msg" << std::endl;
   std::cout << "Msg Code: "<< done_msg.code() << std::endl;
   std::cout << "Msg Size: "<< done_msg.size() << std::endl; 

   std::cout << std::endl;
   std::cout << "Local buffer contents with the Other Message (hex):" << std::endl;
   std::cout << std::hex;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     std::cout << " " << (0xff&((int)mybuf[idx]));
   }
   std::cout << std::dec << std::endl;
   std::cout << std::endl;

   std::cout << "Building View on Same Message" << std::endl;
   OtherMessageView other_view(mybuf);
   std::cout << "Reading From  View" << std::endl;
   std::cout << "Msg Code From View: "<< other_view.code() << std::endl;
   std::cout << "Msg Size From View: "<< other_view.size() << std::endl;
 
   std::cout << std::endl;
   std::cout << "Making a second Other Message with body size of 12 and msg code 10" << std::endl;
   OtherMessageBuilder  second_msg(mybuf, 10, 12);
   std::cout << "Reading from Msg" << std::endl;
   std::cout << "Msg Code: "<< second_msg.code() << std::endl;
   std::cout << "Msg Size: "<< second_msg.size() << std::endl; 

   std::cout << "Setting the second message body to Hello World" << std::endl;
   char *bodyPtr = (char *) second_msg.msgBody();
   strcpy(bodyPtr, "Hello World");
   
   std::cout << std::endl;
   std::cout << "Local buffer contents with the second Other Message (hex):"<< std::endl;
   std::cout << std::hex;
   for (unsigned int idx = 0; idx < sizeof(mybuf); ++idx) {
     std::cout <<" "<< (0xff&((int)mybuf[idx]));
   }
   std::cout << std::dec << std::endl;
   std::cout << std::endl;

   std::cout << "Building View on second Message" << std::endl;
   OtherMessageView second_view(mybuf);
   std::cout << "Reading From  View" << std::endl;
   std::cout << "Msg Code From View: "<< second_view.code() << std::endl;
   std::cout << "Msg Size From View: "<< second_view.size() << std::endl;
 
   std::cout << "Second Message body = " << second_view.msgBody() << std::endl;
}
