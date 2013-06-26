#!/bin/tcsh 


set odir = $1
set iter = $2
echo $odir
set name = `basename $odir`
set jobs = `ls -d $odir/job*/ | wc -l`


     UP: 
        @ alldone1 = 0
        set ii = $jobs

 	while($ii)
        if(`ls -q $odir/job$ii/ |  grep DONE1` != "DONE1" ) then 
         if(`stat -c %s $odir/job$ii/IOUserVariables.root` > 1000 || `stat -c %s $odir/job$ii/IOUserVariables.root` == 0 ) then
          sleep 5
          echo dentro primo if job$ii          
          if(`stat -c %s $odir/job$ii/IOUserVariables.root` > 5000) then
          @ alldone1 = $alldone1 + 1
          touch $odir/job$ii/DONE1
          echo dentro secondo if job$ii          
          else
          echo 1 > $odir/job$ii/DONE 
          rm -f $odir/job$ii/IOUserVariables.root
          bkill -J $name/align$iter\[$ii\]  
                     	 
          echo kill the job $name/align$iter\[$ii\]    
         
          endif
         endif
        else
        echo dentro primo else job$ii          
         @ alldone1 = $alldone1 + 1  
       
        endif
       
        @ ii--    
        end
       
     if($alldone1 != $jobs) then 
        sleep 10
        echo nuovo ciclo
        goto UP
     endif

        
           



