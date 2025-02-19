#!/usr/bin/env perl
$libdir = @ARGV[0];
$moddir = @ARGV[1];
print "fixing library clone at $libdir\n";
`rm $libdir/libgcc_s.so`;
`rm $libdir/libgcc_s.so.1`;
`rm $libdir/liblcg_LFCCatalog.so`;
`rm $libdir/liblcg_GliteCatalog.so`;
`rm $libdir/liblcg_GTCatalog.so`;
`rm $moddir/lcg_LFCCatalog.reg`;
`rm $moddir/lcg_GliteCatalog.reg`;
`rm $moddir/lcg_GTCatalog.reg`;

