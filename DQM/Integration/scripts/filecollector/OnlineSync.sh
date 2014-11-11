#!/bin/zsh
BASE_DIR=/dqmdata/offline/repository/data/OnlineData
X509_CERT_DIR=/etc/grid-security/certificates
X509_USER_PROXY=/dqmdata/auth/proxy.pem

#BASE_DIR=~/DQMWorkArea/dqmdata/repository/data/OnlineData
#X509_USER_PROXY=~/.globus/x509up_u44417
#X509_CERT_DIR=/afs/cern.ch/project/gd/LCG-share2/certificates/

Counter=0
log(){
  echo $(date +"%F %T") \[OnlineSynchronyzer/$$\] $1
}
rcurl(){
  [[ -z $2 ]] && removeTree=0 || removeTree=$2
  if [[ $removeTree -eq 0 ]]
  then
    rootDir=$1 
    remoteDirBase="dqm/online/data/browse"
  else
    [[ ${(w)#${(SI:$removeTree:)1/\// }} -gt 1 ]] && rootDir=${${(s: :SI:$removeTree:)1/\// }[2]} || rootDir=""
    [[ ${(w)#${(SI:$removeTree:)1/\// }} -gt 1 ]] && remoteDirBase="dqm/online/data/browse"/${${(s: :SI:$removeTree:)1/\// }[1]} || remoteDirBase="dqm/online/data/browse"/$1
  fi
  host="https://cmsweb.cern.ch"
  [[ -e $rootDir/index.html ]] && dirTimeStamp=$(date -d "$(curl -A "OnlineSyncDev/0.1" -s -I \
       --capath $X509_CERT_DIR/ \
       --key $X509_USER_PROXY \
       --cert $X509_USER_PROXY \
       $host/$remoteDirBase/$rootDir/ | grep "Last-Modified:.*$" | sed -e "s|Last-Modified: ||" )" +%s) || dirTimeStamp=$(date +%s)
  if [[ ! -e $rootDir/index.html || $dirTimeStamp -gt $(stat $rootDir/index.html -c %Y) ]]
  then 
    echo Fetching $rootDir content
    curl -A "OnlineSyncDev/0.1" -s -R -o $rootDir/index.html --create-dirs \
         --capath $X509_CERT_DIR/ \
         --key $X509_USER_PROXY \
         --cert $X509_USER_PROXY \
         $host/$remoteDirBase/$rootDir/ 
  fi
  dirs=($(<$rootDir/index.html | egrep -oe "<tr><td><a.*</a>" | egrep -o "href='([^']*)'" | egrep -v "\.root" | sed -re "s/(href='\/|\/'$)//g" | sort -r ))
  files=($(<$rootDir/index.html | egrep -oe "<tr><td><a.*</a>" | egrep -o "href='([^']*)'" | egrep  "\.root" |  sed -re "s/(href='\/|'$)//g" | sort -r ))
  for d in $dirs
  do
    rcurl ${d/dqm\/online\/data\/browse\//} $removeTree
  done
  for f in $files
  do
    [[ -e $rootDir/$(basename $f) ]] && fTimeStamp=$(date -d "$(curl -A "OnlineSyncDev/0.1" -s -I \
       --capath $X509_CERT_DIR/ \
       --key $X509_USER_PROXY \
       --cert $X509_USER_PROXY \
       $host/$remoteDirBase/$rootDir/$(basename $f ) | grep "Last-Modified:.*$" | sed -e "s|Last-Modified: ||" )" +%s) || fTimeStamp=$(date +%s)
    if [[ ! -e $rootDir/$(basename $f) || $fTimeStamp -gt $(stat $rootDir/$(basename $f) -c %Y) ]] 
    then
      curl -A "OnlineSyncDev/0.1" -s -R -o $rootDir/$(basename $f) --create-dirs \
         --capath $X509_CERT_DIR/ \
         --key $X509_USER_PROXY \
         --cert $X509_USER_PROXY \
         $host/$f && log "INFO: Successfully downloaded "$PWD/$rootDir/$(basename $f) || \
         log "ERROR: Unable to download "$PWD/$rootDir/$(basename $f)
    fi
    
  done
  
}
while [ 1 ]
do 
  if [[ $Counter -gt 7 ]]
  then
    Counter=0
    #log "INFO: Starting daily full tree synchronization"
    #cd $BASE_DIR/original
    #[[ $(pwd) == $BASE_DIR/original ]] && wget -q -e robots=off --mirror -np -nH --cut-dirs=5 -T120 \
    #   --ca-dir /etc/grid-security/certificates \
    #   --private-key /dqmdata/auth/proxy.pem \
    #   --certificate /dqmdata/auth/proxy.pem \
    #   https://cmsweb.cern.ch/dqm/online/data/browse/Original/ || (log "ERROR: Could not go to $BASE_DIR/original";exit)
    #cd $BASE_DIR/merged
    #[[ $(pwd) == $BASE_DIR/merged ]] && wget -q -e robots=off --mirror -np -nH --cut-dirs=5 -T120 \
    #   --ca-dir /etc/grid-security/certificates \
    #   --private-key /dqmdata/auth/proxy.pem \
    #   --certificate /dqmdata/auth/proxy.pem \
    #   https://cmsweb.cern.ch/dqm/online/data/browse/Merged/ || (log "ERROR: Could not go to $BASE_DIR/Merged";exit)
    #log "INFO: Finished daily full tree synchronization"wget
    continue
  fi
  Counter=$(( $Counter + 1 ))
  cd $BASE_DIR/original
  latestLocalDir=$(find $BASE_DIR/original -maxdepth 1 -type d -exec basename {} \; | sort -n | tail -n 1)
  latestDir=$(curl -A "OnlineSyncDev/0.1" \
    --capath $X509_CERT_DIR --key $X509_USER_PROXY \
    --cert $X509_USER_PROXY \
    https://cmsweb.cern.ch/dqm/online/data/browse/Original/ 2>&1 \
    | egrep -oe "<tr><td><a.*</a>" | egrep -o "'>.*<" | egrep -o "[0-9]{5}xxxx" |sort -r | head -n 1)
  [[ -z $latestDir ]] && continue
  dirs=({${latestLocalDir/xxxx/}..${latestDir/xxxx/}}xxxx)
  #log "INFO: Starting partial synchronization of original files"
  for d in $dirs
  do 
    #log "INFO: Synchronizing  https://cmsweb.cern.ch/dqm/online/data/browse/Original/$d"
    rcurl Original/$d 1
  done
  #log "INFO: Finished partial synchronization of original files"

  cd $BASE_DIR/merged
  latestLocalDir=$(find $BASE_DIR/merged -maxdepth 1 -type d -exec basename {} \; | sort -n | tail -n 1)
  latestDir=$(curl -A "OnlineSyncDev/0.1" \
    --capath $X509_CERT_DIR --key $X509_USER_PROXY \
    --cert $X509_USER_PROXY \
    https://cmsweb.cern.ch/dqm/online/data/browse/Merged/ 2>&1|
    egrep -oe "<tr><td><a.*</a>" | egrep -o "'>.*<" | egrep -o "[0-9]{5}xxxx" |sort -r | head -n 1)
  dirs=({${latestLocalDir/xxxx/}..${latestDir/xxxx/}}xxxx)
  log "INFO: Starting partial synchronization of merged files"
  if [[ X$latestDir != X ]]
  then 
    for d in $dirs
    do
      log "INFO: Synchronizing  https://cmsweb.cern.ch/dqm/online/data/browse/Merged/$d"
      rcurl Merged/$d 1
    done
    log "INFO: Finished partial synchronization of merged files"
  fi
  sleep $(( 3600 * 4 ))
done
