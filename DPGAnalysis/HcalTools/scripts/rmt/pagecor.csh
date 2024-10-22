#!/bin/csh
set refnumber=213224
set WebSite='https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/'
set WebDir='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb'
foreach i (`cat LED_LIST/runlist.tmp.2013-11-12_11_37_01`)
set runnumber=`echo ${i} | awk -F _ '{print $1}'`
set j=`cat LED_LIST/runlist.tmp0.2013-11-12_11_37_01 | grep ${runnumber}`
setenv runtype `echo $j | awk -F - '{print $13}'`
setenv runHTML `echo $j | awk -F - '{print $25}' | awk -F 'href=' '{print $2}'`
setenv runday `echo $j | awk -F - '{print $19}'`
setenv runmonth `echo $j | awk -F - '{print $18}'`
setenv runyear `echo $j | awk -F - '{print $17}'`
setenv runtime `echo $j | awk -F - '{print $20}'`
setenv rundate ${runday}"."${runmonth}"."${runyear}
wget ${runHTML} -O index.html.${runnumber}
setenv runNevents `cat index.html.${runnumber} | tail -n +14 | head -n 1 | awk -F '>' '{print $2}' | awk -F '<' '{print $1}'`
echo ${runnumber} ${j} ${runtype}
touch index_draft.html.${runnumber}
set raw=3
echo '<tr>'>> index_draft.html.${runnumber}
echo '<td class="s1" align="center">'ktemp'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$runnumber'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$runtype'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$runNevents'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$rundate'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$runtime'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">'$refnumber'</td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center"><a href="'$WebSite'/LED_'$runnumber'/MAP.html">LED_'$runnumber'</a></td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center"><a href="'$runHTML'">DetDiag_'$runnumber'</a></td>'>> index_draft.html.${runnumber}
echo '<td class="s'$raw'" align="center">OK</td>'>> index_draft.html.${runnumber}
echo '</tr>'>> index_draft.html.${runnumber}
mv $WebDir/LED_$runnumber/index_draft.html $WebDir/LED_$runnumber/index_draft.html.orig
cp index_draft.html.${runnumber} $WebDir/LED_$runnumber/index_draft.html
end


