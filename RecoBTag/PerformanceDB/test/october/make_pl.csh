rm -f PhysicsPerformance.db
rm -f */PhysicsPerformance.db
rm -f PoolFragment.txt
touch PoolFragment.txt
rm -f BtagFragment.txt
touch BtagFragment.txt

cd mistag
./CREATE.csh MISTAG
cp PhysicsPerformance.db ../
cat Pool_template.MISTAG >> ../PoolFragment.txt
cat Btag_template.MISTAG >> ../BtagFragment.txt
cd ../

cp PhysicsPerformance.db ptrel
cd ptrel
./CREATE.csh PTREL
cat Pool_template.PTREL >> ../PoolFragment.txt
cat Btag_template.PTREL >> ../BtagFragment.txt
cp PhysicsPerformance.db ../
cd ../

cp PhysicsPerformance.db s8
cd s8
./CREATE.csh SYSTEM8
cat Pool_template.SYSTEM8 >> ../PoolFragment.txt
cat Btag_template.SYSTEM8 >> ../BtagFragment.txt
cp PhysicsPerformance.db ../
cd ../
#

cat Pool_pre.fragment PoolFragment.txt Pool_post.fragment > ! PoolBTagPerformanceDBOctober09.py
cat Btag_pre.fragment BtagFragment.txt > ! BTagPerformanceDBOctober09.py
#

