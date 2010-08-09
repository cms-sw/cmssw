#!/bin/bash
if [ -z "$1" ] #|| [ -z "$2" ] || [ -z "$3" ]
then
        echo "Usage: ./configure.sh <user name> [backend server address] [backend server port number]"
        exit
fi

userOld="lxplusUser"
addressOld="backendAddress"
#addressOld="popcon2vm.cern.ch"
#addressOld="webcondvm.cern.ch"
portOld="backendPort"

#update user name          
#sed -i "s/$userOld/${1}/g" *.*
#sed -i "s/$userOld/${1}/g" *.php
sed -i "s/$userOld/${1}/g" *.html*  
#sed -i "s/$userOld/${1}/g" js/*.*

#update server address
#sed -i "s/$addressOld/${2}/g" *.*
#sed -i "s/$addressOld/${2}/g" *.php
sed -i "s/$addressOld/${2}/g" *.html*  
#sed -i "s/$addressOld/${2}/g" js/*.*

#update server address
#sed -i "s/$portOld/${3}/g" *.*
#sed -i "s/$portOld/${3}/g" *.php
sed -i "s/$portOld/${3}/g" *.html*  
#sed -i "s/$portOld/${3}/g" js/*.*
