#!/usr/local/bin/perl

## Usage:
## cat yourtexfile | ./tex2xml > newmaterials.xml  

while (<>)
{
    if(/ Norm. mixture density \[g\/cm\$\^3\$] &\s* ([\d\.]+)\s*/ )
  {
   $density=$1;
    print "\n";
    print "  <CompositeMaterial name=\"$material\" density=\"$density*g/cm3\" symbol=\" \" method=\"mixture by weight\">\n";
    $i=0;
    $tot=0;
    while (($compsname, $compsperc) = each(%comps))
#     while($i<$ncomp)
    {
     print "    <MaterialFraction fraction=\"".$compsperc."\">\n";
     print "      <rMaterial name=\"materials:".$compsname."\"/>\n";
     print "    </MaterialFraction>\n";
     $tot+=$compsperc;
     $i++;
   }  
    print "  </CompositeMaterial>\n";
#    print STDERR "Total fraction is: ".$tot."\n"; 
    if(abs($tot-1) > 0.001 ) 
    {
     die "ERROR: total fraction is ".$tot; 
    }
   $flushed=1; 
 }

if(/Material name:\s*([\w\\]+)/) #new mixture
# OLD:    if(/Material name: ([\w\\]+)/) #new mixture
  {
    $flushed=0;
   # @compsname = ();
    %comps = ();
    $ncomp =0;
    $material = $1; 
    $material =~ s/\\//g;
    #print "New material: ".$material."\n";
    $inmaterial=1;
  }
  if(/^(\s*[0-9]+\s*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)&(.*)/)
  {
    $compsname = $3;
    $compsperc = $7;
    $compsperc =~ s/\s*//g;
    $compsperc /= 100;
    $compsname =~ s/^\s*//;
    $compsname =~ s/\s*$//;
    $compsname =~ s/\\//g;
    $comps{$compsname} += $compsperc;
     $ncomp++;
 }
  if(/\\end{tabular}/)
  { 
   $inmaterial=0;
   $density=-1;
  }
 

}
