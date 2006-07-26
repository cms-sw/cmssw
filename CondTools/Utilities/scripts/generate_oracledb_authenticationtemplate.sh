#! /bin/bash
#---------------------------------------------------------------------
# generate the authentication_template.xml for Read-Only users of official 
# CMS conditions data oracle servers. 
# Note: This script does not cover authentication for private 
# and development databases. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
AUTH_FILE=authentication_template.xml
OWNERS="   
GENERAL
CSC
DT
ECAL
HCAL
PIXEL
PRESH
RPC
STRIP
" 
SERVERLIST="
ORCON
CMS_ORCOFF
CMS_ORCOFF_INT2R
CMS_ORCOFF_VAL
DEVDB10
"
# ------------------------------------------------------------------------
#  
# Parameters: 0
# Returns: 0 on success
# ------------------------------------------------------------------------
generate_authxml() {
 local PASS=***
 /bin/rm -f ${AUTH_FILE}
 /bin/cat > ${AUTH_FILE} <<EOF
<?xml version="1.0" ?>
<connectionlist>
EOF
 for HOST in ${SERVERLIST}
 do
  for OWNER in ${OWNERS}
  do
   local CONNECTSTR=oracle://${HOST}/CMS_COND_${OWNER}
   local READONLYUSER=CMS_COND_${OWNER}
   if [ ${HOST} != 'DEVDB10' ]
   then
      READONLYUSER=${READONLYUSER}'_R'
   fi
/bin/cat >> ${AUTH_FILE} <<EOF
 <connection name="${CONNECTSTR}">
   <parameter name="user" value="${READONLYUSER}"/>
   <parameter name="password" value="${PASS}"/>
 </connection>
EOF
  done
 done
/bin/cat >> ${AUTH_FILE} <<EOF
</connectionlist>
EOF
 return 0
}
print_usage(){
  echo "Usage: `basename $0` "
  echo " Generate authentication_template.xml for CMS oracle services: "
  echo " ORCON,CMS_ORCOFF,CMS_ORCOFF_INT2R,CMS_ORCOFF_VAL "
  exit 0
}
# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------

while getopts ":h" Option
do
  case $Option in
    h ) print_usage
        ;;
    * ) echo "Unimplemented option chosen";;
  esac
done 
shift $(($OPTIND -1))
#list of schema owners
  generate_authxml    
echo "Done! ${AUTH_FILE} generated"
exit 0
