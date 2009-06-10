thisHost=`hostname`
if [[ $thisHost =~ srv-C ]]
then
    source /nfshome0/cmssw2/scripts/setup.sh
else
    if [[ $thisHost =~ fnal.gov ]]
    then
        source /uscmst1/prod/sw/cms/shrc uaf
    fi
fi
